import XCTest
@testable import HermesCore

@MainActor
final class HermesCoreTests: XCTestCase {
    func testEndpointNormalizesRootURL() throws {
        let endpoint = try Endpoint("HTTPS://Mac.Example.TS.NET/")
        XCTAssertEqual(endpoint.baseURL.absoluteString, "https://mac.example.ts.net")
    }

    func testEndpointRejectsUnsafeOriginsAndPaths() {
        let invalid = [
            "http://mac.example.ts.net",
            "https://mac.example.com",
            "https://user:password@mac.example.ts.net",
            "https://mac.example.ts.net/#fragment",
            "https://mac.example.ts.net/?ticket=secret",
            "https://mac.example.ts.net/../api/ws",
            "https://mac.example.ts.net/api/ws",
        ]
        for raw in invalid {
            XCTAssertThrowsError(try Endpoint(raw), raw)
        }
    }

    func testJSONAccessorsAndMissingObjectKey() throws {
        let value: JSON = .object([
            "text": .string("hello"),
            "ok": .bool(true),
            "count": .number(4),
            "items": .array([.null]),
        ])
        XCTAssertEqual(value["text"].string, "hello")
        XCTAssertTrue(value["ok"].bool == true)
        XCTAssertEqual(value["count"].int, 4)
        XCTAssertEqual(value["items"].array, [.null])
        XCTAssertEqual(value["missing"], .null)
        XCTAssertNil(value.string)
        XCTAssertNil(JSON.number(Double(Int.max)).int)
    }

    func testGatewayDispatchesNotificationParamsRatherThanEnvelope() async throws {
        let socket = FakeSocket(messages: [rpcResult(id: "1", result: ["pong": true])])
        let gateway = makeGateway(socket: socket)
        try await gateway.connect(endpoint: try Endpoint("https://mac.example.ts.net"), username: "ada", password: "pw")

        let event = expectation(description: "notification params")
        gateway.onEvent = { params in
            XCTAssertEqual(params["type"].string, "message.delta")
            XCTAssertEqual(params["session_id"].string, "session-1")
            XCTAssertEqual(params["payload"]["text"].string, "Hi")
            XCTAssertNil(params["method"].string)
            event.fulfill()
        }
        socket.enqueue(notification(type: "message.delta", session: "session-1", payload: ["text": "Hi"]))
        await fulfillment(of: [event], timeout: 1)
    }

    func testRPCErrorAndTimeoutCleanPendingCalls() async throws {
        let socket = FakeSocket(messages: [rpcResult(id: "1", result: ["pong": true])])
        let gateway = makeGateway(socket: socket, timeout: 0.03)
        try await gateway.connect(endpoint: try Endpoint("https://mac.example.ts.net"), username: "ada", password: "pw")

        socket.enqueue(rpcError(id: "2", code: -32000, message: "Denied"))
        do {
            _ = try await gateway.request("denied", params: .object([:]))
            XCTFail("Expected JSON-RPC error")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Denied"))
        }

        do {
            _ = try await gateway.request("hang", params: .object([:]))
            XCTFail("Expected timeout")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("timed out"))
        }
        XCTAssertEqual(gateway.pendingRequestCount, 0)
    }

    func testRPCUsesTextWebSocketFrameWithNewlineDelimiter() async throws {
        let socket = FakeSocket(messages: [
            rpcResult(id: "1", result: ["pong": true]),
            rpcResult(id: "2", result: ["accepted": true]),
        ])
        let gateway = makeGateway(socket: socket)
        try await gateway.connect(endpoint: try Endpoint("https://mac.example.ts.net"), username: "ada", password: "pw")
        _ = try await gateway.request("session.list", params: .object([:]))
        XCTAssertTrue(socket.sentTexts.allSatisfy { $0.hasSuffix("\n") })
        XCTAssertTrue(socket.sentTexts.allSatisfy { text in
            guard let data = text.data(using: .utf8),
                  let frame = try? JSONDecoder().decode(JSON.self, from: data) else { return false }
            return frame["jsonrpc"].string == "2.0"
        })
        guard case let .string(text) = WebSocketWire.outgoingMessage(text: "{}\n") else {
            return XCTFail("Gateway WebSocket transport must use a text message")
        }
        XCTAssertEqual(text, "{}\n")
    }

    func testSupersededConnectCannotPostPasswordOrCloseNewSocket() async throws {
        let socket = FakeSocket(messages: [rpcResult(id: "1", result: ["pong": true])])
        let factory = FakeSocketFactory(sockets: [socket])
        let oldEndpoint = try Endpoint("https://old.example.ts.net")
        let newEndpoint = try Endpoint("https://new.example.ts.net")
        let gateway = makeGateway(factory: factory, stub: { request in
            if request.url?.host == "old.example.ts.net", request.url?.path == "/api/auth/providers" {
                return .json(["providers": [["name": "password", "display_name": "Password", "supports_password": true]]], delay: 0.08)
            }
            return Self.authResponse(for: request)
        })
        let old = Task { @MainActor in try await gateway.connect(endpoint: oldEndpoint, username: "old", password: "old-password") }
        for _ in 0 ..< 20 where !HTTPStub.requests.contains(where: { $0.url?.host == "old.example.ts.net" }) {
            await Task.yield()
        }
        try await gateway.connect(endpoint: newEndpoint, username: "new", password: "new-password")
        do {
            _ = try await old.value
            XCTFail("Superseded attempt must fail")
        } catch {}
        XCTAssertFalse(socket.cancelled)
        XCTAssertFalse(HTTPStub.requests.contains(where: { $0.url?.host == "old.example.ts.net" && $0.url?.path == "/auth/password-login" }))
        XCTAssertTrue(HTTPStub.requests.contains(where: { $0.url?.host == "new.example.ts.net" && $0.url?.path == "/auth/password-login" }))
    }

    func testDisconnectFailsPendingRequestAndStaleSocketCannotDisconnectNewOne() async throws {
        let first = FakeSocket(messages: [rpcResult(id: "1", result: ["pong": true])])
        let second = FakeSocket(messages: [rpcResult(id: "2", result: ["pong": true])])
        let factory = FakeSocketFactory(sockets: [first, second])
        let gateway = makeGateway(factory: factory, timeout: 2)
        let endpoint = try Endpoint("https://mac.example.ts.net")
        try await gateway.connect(endpoint: endpoint, username: "ada", password: "pw")
        try await gateway.reconnect()
        first.failReceive(with: URLError(.networkConnectionLost))
        try await Task.sleep(for: .milliseconds(30))

        XCTAssertFalse(second.cancelled)
        let request = Task { @MainActor in try await gateway.request("hang", params: .object([:])) }
        for _ in 0 ..< 10 where gateway.pendingRequestCount == 0 { await Task.yield() }
        XCTAssertEqual(gateway.pendingRequestCount, 1)
        gateway.disconnect()
        do {
            _ = try await request.value
            XCTFail("Expected disconnect")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("disconnected"))
        }
        XCTAssertEqual(gateway.pendingRequestCount, 0)
    }

    func testReconnectMintsFreshTicket() async throws {
        let first = FakeSocket(messages: [rpcResult(id: "1", result: ["pong": true])])
        let second = FakeSocket(messages: [rpcResult(id: "2", result: ["pong": true])])
        let factory = FakeSocketFactory(sockets: [first, second])
        let gateway = makeGateway(factory: factory)
        try await gateway.connect(endpoint: try Endpoint("https://mac.example.ts.net"), username: "ada", password: "pw")
        try await gateway.reconnect()
        XCTAssertEqual(factory.urls.map { URLComponents(url: $0, resolvingAgainstBaseURL: false)?.queryItems?.first(where: { $0.name == "ticket" })?.value }, ["ticket-1", "ticket-2"])
    }

    func testRedirectIsNotFollowedOrSentToOtherOrigin() async throws {
        let socket = FakeSocket(messages: [rpcResult(id: "1", result: ["pong": true])])
        let protocolStub: (URLRequest) -> HTTPStub.Response = { request in
            if request.url?.path == "/redirect" {
                return HTTPStub.Response(status: 302, headers: ["Location": "https://attacker.example.ts.net/steal"], body: Data())
            }
            return Self.authResponse(for: request)
        }
        let gateway = makeGateway(socket: socket, stub: protocolStub)
        try await gateway.connect(endpoint: try Endpoint("https://mac.example.ts.net"), username: "ada", password: "pw")
        await XCTAssertThrowsErrorAsync(try await gateway.http("/redirect"))
        XCTAssertEqual(HTTPStub.requests.map(\.url?.host), ["mac.example.ts.net", "mac.example.ts.net", "mac.example.ts.net", "mac.example.ts.net", "mac.example.ts.net"])
    }

    func testHTTPStatusFailureIsSurfaced() async throws {
        let socket = FakeSocket(messages: [rpcResult(id: "1", result: ["pong": true])])
        let gateway = makeGateway(socket: socket, stub: { request in
            if request.url?.path == "/failure" {
                return HTTPStub.Response(status: 401, headers: ["Content-Type": "application/json"], body: Data("{\"detail\":\"Unauthorized\"}".utf8))
            }
            return Self.authResponse(for: request)
        })
        try await gateway.connect(endpoint: try Endpoint("https://mac.example.ts.net"), username: "ada", password: "pw")
        do {
            _ = try await gateway.http("/failure")
            XCTFail("Expected HTTP status failure")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("401"))
        }
    }

    func testSameHostDifferentPortDoesNotAttachPreviousCookie() async throws {
        let first = FakeSocket(messages: [rpcResult(id: "1", result: ["pong": true])])
        let second = FakeSocket(messages: [rpcResult(id: "2", result: ["pong": true])])
        let gateway = makeGateway(factory: FakeSocketFactory(sockets: [first, second]), stub: { request in
            if request.url?.port == 9443, request.url?.path == "/api/auth/providers" {
                XCTAssertNil(request.value(forHTTPHeaderField: "Cookie"))
            }
            return Self.authResponse(for: request)
        })
        try await gateway.connect(endpoint: try Endpoint("https://mac.example.ts.net:8443"), username: "a", password: "p")
        try await gateway.connect(endpoint: try Endpoint("https://mac.example.ts.net:9443"), username: "b", password: "p")
    }

    func testRotatedCookieOnGETPersistsForReconnect() async throws {
        let first = FakeSocket(messages: [rpcResult(id: "1", result: ["pong": true])])
        let second = FakeSocket(messages: [rpcResult(id: "2", result: ["pong": true])])
        var sawRotated = false
        let gateway = makeGateway(factory: FakeSocketFactory(sockets: [first, second]), stub: { request in
            if request.url?.path == "/rotate" {
                return .json(["ok": true], headers: ["Set-Cookie": "__Host-hermes_session_at=rotated; Path=/; Secure; HttpOnly"])
            }
            if request.url?.path == "/api/auth/me", request.value(forHTTPHeaderField: "Cookie")?.contains("rotated") == true { sawRotated = true }
            return Self.authResponse(for: request)
        })
        let endpoint = try Endpoint("https://mac.example.ts.net")
        try await gateway.connect(endpoint: endpoint, username: "a", password: "p")
        _ = try await gateway.http("/rotate")
        try await gateway.reconnect()
        XCTAssertTrue(sawRotated)
    }

    func testRotatedCookieOnHTTPErrorPersistsForReconnect() async throws {
        let first = FakeSocket(messages: [rpcResult(id: "1", result: ["pong": true])])
        let second = FakeSocket(messages: [rpcResult(id: "2", result: ["pong": true])])
        var sawRotated = false
        let gateway = makeGateway(factory: FakeSocketFactory(sockets: [first, second]), stub: { request in
            if request.url?.path == "/conflict" {
                return HTTPStub.Response(status: 409, headers: ["Set-Cookie": "__Host-hermes_session_at=rotated409; Path=/; Secure; HttpOnly"], body: Data("{}".utf8))
            }
            if request.url?.path == "/api/auth/me", request.value(forHTTPHeaderField: "Cookie")?.contains("rotated409") == true { sawRotated = true }
            return Self.authResponse(for: request)
        })
        let endpoint = try Endpoint("https://mac.example.ts.net")
        try await gateway.connect(endpoint: endpoint, username: "a", password: "p")
        await XCTAssertThrowsErrorAsync(try await gateway.http("/conflict"))
        try await gateway.reconnect()
        XCTAssertTrue(sawRotated)
    }

    func testDelayedPasswordLoginAfterSignOutCannotPersistCookie() async throws {
        let store = MemoryCredentialStore()
        let endpoint = try Endpoint("https://old.example.ts.net")
        let gateway = makeGateway(socket: FakeSocket(), store: store, stub: { request in
            if request.url?.path == "/auth/password-login" {
                return .json(["ok": true], headers: ["Set-Cookie": "__Host-hermes_session_at=stale; Path=/; Secure; HttpOnly"], delay: 0.08)
            }
            return Self.authResponse(for: request)
        })
        let pending = Task { @MainActor in try await gateway.connect(endpoint: endpoint, username: "a", password: "p") }
        for _ in 0 ..< 20 where !HTTPStub.requests.contains(where: { $0.url?.path == "/auth/password-login" }) { await Task.yield() }
        gateway.signOut()
        _ = try? await pending.value
        XCTAssertTrue(store.cookies(for: endpoint.baseURL).isEmpty)
    }

    func testDelayedPasswordLoginAfterNewConnectCannotPersistCookie() async throws {
        let store = MemoryCredentialStore()
        let oldEndpoint = try Endpoint("https://old.example.ts.net")
        let newEndpoint = try Endpoint("https://new.example.ts.net")
        let socket = FakeSocket(messages: [rpcResult(id: "1", result: ["pong": true])])
        let gateway = makeGateway(factory: FakeSocketFactory(sockets: [socket]), store: store, stub: { request in
            if request.url?.host == "old.example.ts.net", request.url?.path == "/auth/password-login" {
                return .json(["ok": true], headers: ["Set-Cookie": "__Host-hermes_session_at=stale; Path=/; Secure; HttpOnly"], delay: 0.08)
            }
            return Self.authResponse(for: request)
        })
        let pending = Task { @MainActor in try await gateway.connect(endpoint: oldEndpoint, username: "a", password: "p") }
        for _ in 0 ..< 20 where !HTTPStub.requests.contains(where: { $0.url?.host == "old.example.ts.net" && $0.url?.path == "/auth/password-login" }) { await Task.yield() }
        try await gateway.connect(endpoint: newEndpoint, username: "b", password: "p")
        _ = try? await pending.value
        XCTAssertTrue(store.cookies(for: oldEndpoint.baseURL).isEmpty)
        XCTAssertFalse(store.cookies(for: newEndpoint.baseURL).isEmpty)
    }

    func testRedirectDelegateAlwaysCancelsTargetRequest() {
        let guardDelegate = RedirectGuard()
        let session = URLSession(configuration: .ephemeral)
        let original = URLRequest(url: URL(string: "https://mac.example.ts.net/a")!)
        let task = session.dataTask(with: original)
        let response = HTTPURLResponse(url: original.url!, statusCode: 302, httpVersion: "HTTP/1.1", headerFields: ["Location": "https://evil.example.ts.net"])!
        var redirected: URLRequest? = original
        guardDelegate.urlSession(session, task: task, willPerformHTTPRedirection: response, newRequest: URLRequest(url: URL(string: "https://evil.example.ts.net")!)) { redirected = $0 }
        XCTAssertNil(redirected)
    }
}

private extension HermesCoreTests {
    func makeGateway(socket: FakeSocket? = nil, factory: FakeSocketFactory? = nil, timeout: TimeInterval = 1, store: MemoryCredentialStore? = nil, stub: ((URLRequest) -> HTTPStub.Response)? = nil) -> Gateway {
        let socketFactory = factory ?? FakeSocketFactory(sockets: [socket ?? FakeSocket()])
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [HTTPStub.self]
        HTTPStub.responder = stub ?? Self.authResponse
        HTTPStub.reset()
        return Gateway(sessionConfiguration: configuration, socketFactory: socketFactory, credentialStore: store ?? MemoryCredentialStore(), requestTimeout: timeout)
    }

    static func authResponse(for request: URLRequest) -> HTTPStub.Response {
        switch request.url?.path {
        case "/api/auth/providers":
            return .json(["providers": [["name": "password", "display_name": "Password", "supports_password": true]]])
        case "/auth/password-login":
            return .json(["ok": true], headers: ["Set-Cookie": "__Host-hermes_session_at=access; Path=/; Secure; HttpOnly"])
        case "/api/auth/me":
            return .json(["user_id": "user-1", "provider": "password"])
        case "/api/auth/ws-ticket":
            HTTPStub.ticketCount += 1
            return .json(["ticket": "ticket-\(HTTPStub.ticketCount)", "ttl_seconds": 30])
        default:
            return .json([:])
        }
    }

    func rpcResult(id: String, result: [String: Any]) -> Data {
        try! JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "id": id, "result": result])
    }

    func rpcError(id: String, code: Int, message: String) -> Data {
        try! JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "id": id, "error": ["code": code, "message": message]])
    }

    func notification(type: String, session: String, payload: [String: Any]) -> Data {
        try! JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "method": "event", "params": ["type": type, "session_id": session, "payload": payload]])
    }
}

private final class HTTPStub: URLProtocol {
    struct Response {
        let status: Int
        let headers: [String: String]
        let body: Data
        let delay: TimeInterval
        init(status: Int, headers: [String: String], body: Data, delay: TimeInterval = 0) {
            self.status = status
            self.headers = headers
            self.body = body
            self.delay = delay
        }
        static func json(_ object: Any, headers: [String: String] = [:], delay: TimeInterval = 0) -> Response {
            Response(status: 200, headers: ["Content-Type": "application/json"].merging(headers, uniquingKeysWith: { _, new in new }), body: try! JSONSerialization.data(withJSONObject: object), delay: delay)
        }
    }

    static var responder: (URLRequest) -> Response = { _ in .json([:]) }
    static var ticketCount = 0
    static var requests: [URLRequest] = []

    static func reset() { requests = []; ticketCount = 0 }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        HTTPStub.requests.append(request)
        let response = HTTPStub.responder(request)
        let complete = { [weak self] in
            guard let self, let client = self.client else { return }
            let urlResponse = HTTPURLResponse(url: self.request.url!, statusCode: response.status, httpVersion: "HTTP/1.1", headerFields: response.headers)!
            client.urlProtocol(self, didReceive: urlResponse, cacheStoragePolicy: .notAllowed)
            client.urlProtocol(self, didLoad: response.body)
            client.urlProtocolDidFinishLoading(self)
        }
        if response.delay > 0 { DispatchQueue.global().asyncAfter(deadline: .now() + response.delay, execute: complete) } else { complete() }
    }
    override func stopLoading() {}
}

private final class FakeSocketFactory: GatewayWebSocketFactory {
    private var sockets: [FakeSocket]
    private(set) var urls: [URL] = []
    init(sockets: [FakeSocket]) { self.sockets = sockets }
    func make(url: URL) -> GatewayWebSocket {
        urls.append(url)
        return sockets.removeFirst()
    }
}

private final class FakeSocket: GatewayWebSocket {
    private var messages: [Data]
    private var waiters: [CheckedContinuation<Data, Error>] = []
    private var terminalError: Error?
    private(set) var sentTexts: [String] = []
    private(set) var cancelled = false
    init(messages: [Data] = []) { self.messages = messages }
    func send(_ text: String) async throws { sentTexts.append(text) }
    func receive() async throws -> Data {
        if let message = messages.first { messages.removeFirst(); return message }
        if let error = terminalError { throw error }
        return try await withCheckedThrowingContinuation { waiters.append($0) }
    }
    func cancel() { cancelled = true; failReceive(with: GatewayError.disconnected) }
    func enqueue(_ data: Data) { if let waiter = waiters.popLast() { waiter.resume(returning: data) } else { messages.append(data) } }
    func failReceive(with error: Error) { terminalError = error; waiters.forEach { $0.resume(throwing: error) }; waiters = [] }
}

private final class MemoryCredentialStore: CredentialStoring {
    private var cookiesByOrigin: [String: [HTTPCookie]] = [:]
    func save(_ cookies: [HTTPCookie], for origin: URL) throws { cookiesByOrigin[key(origin)] = cookies }
    func load(for origin: URL) throws -> [HTTPCookie] { cookiesByOrigin[key(origin)] ?? [] }
    func clear(for origin: URL) throws { cookiesByOrigin[key(origin)] = [] }
    func cookies(for origin: URL) -> [HTTPCookie] { cookiesByOrigin[key(origin)] ?? [] }
    private func key(_ origin: URL) -> String { origin.absoluteString }
}

private extension XCTestCase {
    func XCTAssertThrowsErrorAsync<T>(_ expression: @autoclosure () async throws -> T, file: StaticString = #filePath, line: UInt = #line) async {
        do { _ = try await expression(); XCTFail("Expected error", file: file, line: line) } catch {}
    }
}
