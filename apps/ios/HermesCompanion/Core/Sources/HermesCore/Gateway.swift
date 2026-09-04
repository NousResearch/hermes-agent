import Foundation

public enum GatewayError: LocalizedError, Sendable {
    case notConnected
    case disconnected
    case requestTimedOut
    case invalidResponse(String)
    case httpStatus(Int)
    case rpc(code: Int, message: String)
    case unsupportedSignIn
    case invalidPath

    public var errorDescription: String? {
        switch self {
        case .notConnected: return "Hermes is not connected."
        case .disconnected: return "The Hermes connection was disconnected."
        case .requestTimedOut: return "The Hermes request timed out."
        case let .invalidResponse(reason): return "Invalid Hermes response: \(reason)"
        case let .httpStatus(status): return "Hermes returned HTTP \(status)."
        case let .rpc(code, message): return "Hermes RPC error \(code): \(message)"
        case .unsupportedSignIn: return "This Hermes server does not offer exactly one password sign-in provider."
        case .invalidPath: return "Only absolute same-origin API paths are allowed."
        }
    }
}

protocol GatewayWebSocket: AnyObject {
    func send(_ text: String) async throws
    func receive() async throws -> Data
    func cancel()
}

protocol GatewayWebSocketFactory: AnyObject {
    func make(url: URL) -> GatewayWebSocket
}

private final class URLSessionWebSocketFactory: GatewayWebSocketFactory {
    private let session: URLSession
    init(session: URLSession) { self.session = session }
    func make(url: URL) -> GatewayWebSocket { URLSessionWebSocket(session.webSocketTask(with: url)) }
}

enum WebSocketWire {
    static func outgoingMessage(text: String) -> URLSessionWebSocketTask.Message { .string(text) }
}

private final class URLSessionWebSocket: GatewayWebSocket {
    private let task: URLSessionWebSocketTask
    init(_ task: URLSessionWebSocketTask) { self.task = task; task.resume() }
    func send(_ text: String) async throws { try await task.send(WebSocketWire.outgoingMessage(text: text)) }
    func receive() async throws -> Data {
        switch try await task.receive() {
        case let .data(data): return data
        case let .string(string): return Data(string.utf8)
        @unknown default: throw GatewayError.invalidResponse("unsupported WebSocket message")
        }
    }
    func cancel() { task.cancel(with: .goingAway, reason: nil) }
}

/// JSON-RPC and authenticated HTTP transport for the private Hermes service.
@MainActor
public final class Gateway {
    public var onEvent: ((JSON) -> Void)?
    public var onDisconnect: ((Error) -> Void)?

    private struct PendingRequest {
        let continuation: CheckedContinuation<JSON, Error>
        let timeout: Task<Void, Never>
    }

    private let session: URLSession
    private let socketFactory: GatewayWebSocketFactory
    private let credentialStore: CredentialStoring
    private let requestTimeout: TimeInterval
    private var endpoint: Endpoint?
    private var socket: GatewayWebSocket?
    private var receiveTask: Task<Void, Never>?
    private var generation = 0
    private var nextRequestID = 0
    private var pending: [String: PendingRequest] = [:]
    private var activeCookies: [HTTPCookie] = []

    public init() {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpShouldSetCookies = false
        configuration.httpCookieStorage = nil
        let session = URLSession(configuration: configuration, delegate: RedirectGuard(), delegateQueue: nil)
        self.session = session
        self.socketFactory = URLSessionWebSocketFactory(session: session)
        self.credentialStore = CredentialStore()
        self.requestTimeout = 15
    }

    init(
        sessionConfiguration: URLSessionConfiguration,
        socketFactory: GatewayWebSocketFactory,
        credentialStore: CredentialStoring,
        requestTimeout: TimeInterval
    ) {
        sessionConfiguration.httpShouldSetCookies = false
        sessionConfiguration.httpCookieStorage = nil
        self.session = URLSession(configuration: sessionConfiguration, delegate: RedirectGuard(), delegateQueue: nil)
        self.socketFactory = socketFactory
        self.credentialStore = credentialStore
        self.requestTimeout = requestTimeout
    }

    public func connect(endpoint: Endpoint, username: String, password: String) async throws {
        closeConnection(error: GatewayError.disconnected, notify: false)
        self.endpoint = endpoint
        let operationGeneration = generation
        do {
            try restoreCookies(for: endpoint)
            let provider = try await passwordProvider(generation: operationGeneration, endpoint: endpoint)
            try requireActive(operationGeneration, endpoint: endpoint)
            let login = try await sendHTTP("/auth/password-login", method: "POST", body: .object([
                "provider": .string(provider),
                "username": .string(username),
                "password": .string(password),
            ]), generation: operationGeneration, endpoint: endpoint)
            guard login["ok"].bool == true else {
                throw GatewayError.invalidResponse("password login did not return ok")
            }
            try await verifyIdentity(generation: operationGeneration, endpoint: endpoint)
            try await openFreshSocket(generation: operationGeneration, endpoint: endpoint)
            try requireActive(operationGeneration, endpoint: endpoint)
            try saveCookies(for: endpoint)
        } catch {
            if isActive(operationGeneration, endpoint: endpoint) {
                closeConnection(error: error, notify: false)
            }
            throw error
        }
    }

    public func reconnect() async throws {
        guard let endpoint else { throw GatewayError.notConnected }
        closeConnection(error: GatewayError.disconnected, notify: false)
        let operationGeneration = generation
        do {
            try restoreCookies(for: endpoint)
            guard !activeCookies.isEmpty else { throw GatewayError.notConnected }
            try await verifyIdentity(generation: operationGeneration, endpoint: endpoint)
            try await openFreshSocket(generation: operationGeneration, endpoint: endpoint)
            try requireActive(operationGeneration, endpoint: endpoint)
            try saveCookies(for: endpoint)
        } catch {
            if isActive(operationGeneration, endpoint: endpoint) {
                closeConnection(error: error, notify: false)
            }
            throw error
        }
    }

    public func request(_ method: String, params: JSON) async throws -> JSON {
        guard let socket else { throw GatewayError.notConnected }
        nextRequestID += 1
        let id = String(nextRequestID)
        let payload = String(decoding: try JSONEncoder().encode(JSON.object([
            "jsonrpc": .string("2.0"),
            "id": .string(id),
            "method": .string(method),
            "params": params,
        ])), as: UTF8.self) + "\n"
        let requestGeneration = generation
        return try await withCheckedThrowingContinuation { continuation in
            let timeout = Task { [weak self] in
                let nanoseconds = UInt64((self?.requestTimeout ?? 15) * 1_000_000_000)
                try? await Task.sleep(nanoseconds: nanoseconds)
                guard !Task.isCancelled else { return }
                self?.timeoutRequest(id)
            }
            pending[id] = PendingRequest(continuation: continuation, timeout: timeout)
            Task { [weak self, weak socket] in
                do {
                    guard let socket else { throw GatewayError.disconnected }
                    guard self?.generation == requestGeneration, self?.socket === socket else {
                        throw GatewayError.disconnected
                    }
                    try await socket.send(payload)
                } catch {
                    self?.completeRequest(id, result: .failure(error))
                }
            }
        }
    }

    public func http(
        _ path: String,
        method: String = "GET",
        body: JSON? = nil,
        query: [String: String] = [:]
    ) async throws -> JSON {
        guard let endpoint else { throw GatewayError.notConnected }
        return try await sendHTTP(path, method: method, body: body, query: query, generation: generation, endpoint: endpoint)
    }

    public func disconnect() {
        closeConnection(error: GatewayError.disconnected, notify: false)
    }

    public func signOut() {
        disconnect()
        guard let endpoint else { return }
        do { try credentialStore.clear(for: endpoint.baseURL) } catch { onDisconnect?(error) }
        activeCookies = []
    }

    var pendingRequestCount: Int { pending.count }

    private func passwordProvider(generation: Int, endpoint: Endpoint) async throws -> String {
        let response = try await sendHTTP("/api/auth/providers", generation: generation, endpoint: endpoint)
        guard let providers = response["providers"].array else {
            throw GatewayError.invalidResponse("providers is missing")
        }
        let passwordProviders = providers.compactMap { provider -> String? in
            guard provider["supports_password"].bool == true,
                  let name = provider["name"].string?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !name.isEmpty else { return nil }
            return name
        }
        guard passwordProviders.count == 1 else { throw GatewayError.unsupportedSignIn }
        return passwordProviders[0]
    }

    private func verifyIdentity(generation: Int, endpoint: Endpoint) async throws {
        let identity = try await sendHTTP("/api/auth/me", generation: generation, endpoint: endpoint)
        guard let userID = identity["user_id"].string, !userID.isEmpty,
              let provider = identity["provider"].string, !provider.isEmpty else {
            throw GatewayError.invalidResponse("identity response is malformed")
        }
    }

    private func openFreshSocket(generation: Int, endpoint: Endpoint) async throws {
        let ticketResponse = try await sendHTTP("/api/auth/ws-ticket", method: "POST", generation: generation, endpoint: endpoint)
        guard let ticket = ticketResponse["ticket"].string, !ticket.isEmpty,
              let ttl = ticketResponse["ttl_seconds"].int, ttl > 0 else {
            throw GatewayError.invalidResponse("WebSocket ticket response is malformed")
        }
        let url = try websocketURL(endpoint: endpoint, ticket: ticket)
        try requireActive(generation, endpoint: endpoint)
        let connectionGeneration = generation
        let socket = socketFactory.make(url: url)
        self.socket = socket
        startReceiveLoop(socket: socket, generation: connectionGeneration)
        let ping = try await request("ping", params: .object([:]))
        try requireActive(generation, endpoint: endpoint)
        guard ping["pong"].bool == true else {
            throw GatewayError.invalidResponse("ping did not return pong")
        }
    }

    private func startReceiveLoop(socket: GatewayWebSocket, generation: Int) {
        receiveTask?.cancel()
        receiveTask = Task { [weak self, weak socket] in
            guard let self, let socket else { return }
            do {
                while !Task.isCancelled {
                    let data = try await socket.receive()
                    guard !Task.isCancelled else { return }
                    self.handleIncoming(data, socket: socket, generation: generation)
                }
            } catch is CancellationError {
                // Explicit cancellation is not an unexpected disconnect.
            } catch {
                self.handleSocketFailure(error, socket: socket, generation: generation)
            }
        }
    }

    private func handleIncoming(_ data: Data, socket: GatewayWebSocket, generation: Int) {
        guard generation == self.generation, self.socket === socket else { return }
        guard let frame = try? JSONDecoder().decode(JSON.self, from: data) else { return }
        if frame["method"].string == "event" {
            // The gateway's application event is the JSON-RPC params object, never the envelope.
            onEvent?(frame["params"])
            return
        }
        guard let id = frame["id"].string else { return }
        if let error = frame["error"].object {
            let code = error["code"]?.int ?? -32000
            let message = error["message"]?.string ?? "Unknown server error"
            completeRequest(id, result: .failure(GatewayError.rpc(code: code, message: message)))
        } else if let result = frame.object?["result"] {
            completeRequest(id, result: .success(result))
        } else {
            completeRequest(id, result: .failure(GatewayError.invalidResponse("RPC response has no result or error")))
        }
    }

    private func handleSocketFailure(_ error: Error, socket: GatewayWebSocket, generation: Int) {
        guard generation == self.generation, self.socket === socket else { return }
        closeConnection(error: error, notify: true)
    }

    private func timeoutRequest(_ id: String) {
        completeRequest(id, result: .failure(GatewayError.requestTimedOut))
    }

    private func completeRequest(_ id: String, result: Result<JSON, Error>) {
        guard let pendingRequest = pending.removeValue(forKey: id) else { return }
        pendingRequest.timeout.cancel()
        pendingRequest.continuation.resume(with: result)
    }

    private func closeConnection(error: Error, notify: Bool) {
        generation &+= 1
        receiveTask?.cancel()
        receiveTask = nil
        socket?.cancel()
        socket = nil
        let openRequests = pending
        pending.removeAll()
        for request in openRequests.values {
            request.timeout.cancel()
            request.continuation.resume(throwing: error)
        }
        if notify { onDisconnect?(error) }
    }

    private func sendHTTP(
        _ path: String,
        method: String = "GET",
        body: JSON? = nil,
        query: [String: String] = [:],
        generation: Int,
        endpoint: Endpoint
    ) async throws -> JSON {
        try requireActive(generation, endpoint: endpoint)
        let request = try makeHTTPRequest(path: path, method: method, body: body, query: query, endpoint: endpoint)
        let (data, response) = try await session.data(for: request)
        try requireActive(generation, endpoint: endpoint)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw GatewayError.invalidResponse("missing HTTP response")
        }
        captureCookies(from: httpResponse, requestURL: request.url)
        try credentialStore.save(activeCookies, for: endpoint.baseURL)
        guard (200 ... 299).contains(httpResponse.statusCode) else {
            throw GatewayError.httpStatus(httpResponse.statusCode)
        }
        do {
            return try JSONDecoder().decode(JSON.self, from: data)
        } catch {
            throw GatewayError.invalidResponse("body is not JSON")
        }
    }

    private func makeHTTPRequest(
        path: String,
        method: String,
        body: JSON?,
        query: [String: String],
        endpoint: Endpoint
    ) throws -> URLRequest {
        guard path.hasPrefix("/"), !path.hasPrefix("//"),
              !path.contains("\\"), !path.split(separator: "/").contains("..") else {
            throw GatewayError.invalidPath
        }
        var components = URLComponents(url: endpoint.baseURL, resolvingAgainstBaseURL: false)
        components?.path = path
        components?.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) }
        guard let url = components?.url,
              url.scheme == endpoint.baseURL.scheme, url.host == endpoint.baseURL.host,
              url.port == endpoint.baseURL.port else { throw GatewayError.invalidPath }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        for (name, value) in HTTPCookie.requestHeaderFields(with: activeCookies) {
            request.setValue(value, forHTTPHeaderField: name)
        }
        if let body {
            request.httpBody = try JSONEncoder().encode(body)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return request
    }

    private func websocketURL(endpoint: Endpoint, ticket: String) throws -> URL {
        var components = URLComponents(url: endpoint.baseURL, resolvingAgainstBaseURL: false)
        components?.scheme = "wss"
        components?.path = "/api/ws"
        components?.queryItems = [URLQueryItem(name: "ticket", value: ticket)]
        guard let url = components?.url else { throw GatewayError.invalidResponse("could not make WebSocket URL") }
        return url
    }

    private func restoreCookies(for endpoint: Endpoint) throws {
        activeCookies = try credentialStore.load(for: endpoint.baseURL)
    }

    private func saveCookies(for endpoint: Endpoint) throws {
        try credentialStore.save(activeCookies, for: endpoint.baseURL)
    }

    private func sessionCookies(for endpoint: Endpoint) -> [HTTPCookie] {
        activeCookies
    }

    private func captureCookies(from response: HTTPURLResponse, requestURL: URL?) {
        guard let requestURL else { return }
        let headers = Dictionary(uniqueKeysWithValues: response.allHeaderFields.compactMap { key, value -> (String, String)? in
            guard let key = key as? String, let value = value as? String else { return nil }
            return (key, value)
        })
        for cookie in HTTPCookie.cookies(withResponseHeaderFields: headers, for: requestURL) {
            activeCookies.removeAll { $0.name == cookie.name && $0.domain == cookie.domain && $0.path == cookie.path }
            if cookie.expiresDate.map({ $0 > Date() }) ?? true { activeCookies.append(cookie) }
        }
    }

    private func isActive(_ generation: Int, endpoint: Endpoint) -> Bool {
        self.generation == generation && self.endpoint == endpoint
    }

    private func requireActive(_ generation: Int, endpoint: Endpoint) throws {
        guard isActive(generation, endpoint: endpoint) else { throw GatewayError.disconnected }
    }
}

/// Keeps URLSession's standard certificate validation while refusing every redirect.
/// The app has no redirect-based flow, so a redirect is an authentication boundary failure.
final class RedirectGuard: NSObject, URLSessionTaskDelegate {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}
