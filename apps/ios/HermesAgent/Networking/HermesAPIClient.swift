import Foundation

/// REST client for the Hermes API Server (the `gateway/platforms/api_server.py`
/// surface). Handles the non-streaming endpoints; streaming lives in `SSEClient`.
actor HermesAPIClient {
    private var connection: Connection
    private let session: URLSession

    init(connection: Connection) {
        self.connection = connection
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.waitsForConnectivity = true
        self.session = URLSession(configuration: config)
    }

    func update(connection: Connection) {
        self.connection = connection
    }

    // MARK: - Health

    func ping() async throws {
        _ = try await get("/health", decode: HealthResponse.self)
    }

    struct HealthResponse: Codable { let status: String? }

    // MARK: - Models & Skills

    func models() async throws -> [ModelInfo] {
        try await get("/v1/models", decode: ModelListResponse.self).data
    }

    func skills() async throws -> [SkillInfo] {
        // The server returns { "data": [...] } or a bare array depending on version.
        let data = try await rawGet("/v1/skills")
        if let env = try? JSONDecoder().decode(SkillListEnvelope.self, from: data) {
            return env.data
        }
        return (try? JSONDecoder().decode([SkillInfo].self, from: data)) ?? []
    }

    private struct SkillListEnvelope: Codable { let data: [SkillInfo] }

    // MARK: - Sessions

    func sessions(limit: Int = 50, offset: Int = 0, source: String? = nil) async throws -> [HermesSession] {
        var path = "/api/sessions?limit=\(limit)&offset=\(offset)"
        if let source { path += "&source=\(source)" }
        return try await get(path, decode: SessionListResponse.self).data
    }

    func createSession(title: String? = nil, model: String? = nil) async throws -> HermesSession {
        var body: [String: Any] = [:]
        if let title { body["title"] = title }
        if let model { body["model"] = model }
        return try await post("/api/sessions", json: body, decode: SessionEnvelope.self).session
    }

    func deleteSession(id: String) async throws {
        _ = try await request(method: "DELETE", path: "/api/sessions/\(id)", body: nil)
    }

    func renameSession(id: String, title: String) async throws {
        _ = try await request(method: "PATCH", path: "/api/sessions/\(id)",
                              body: try JSONSerialization.data(withJSONObject: ["title": title]))
    }

    func messages(sessionId: String) async throws -> [ChatMessage] {
        try await get("/api/sessions/\(sessionId)/messages", decode: MessageListResponse.self).data
    }

    // MARK: - HTTP plumbing

    private func makeRequest(method: String, path: String, body: Data?) throws -> URLRequest {
        guard connection.isValid, let base = connection.url else { throw HermesError.notConfigured }
        guard let url = URL(string: path, relativeTo: base) else { throw HermesError.badURL }
        var req = URLRequest(url: url)
        req.httpMethod = method
        if !connection.apiKey.isEmpty {
            req.setValue("Bearer \(connection.apiKey)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            req.httpBody = body
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        return req
    }

    @discardableResult
    private func request(method: String, path: String, body: Data?) async throws -> Data {
        let req = try makeRequest(method: method, path: path, body: body)
        do {
            let (data, response) = try await session.data(for: req)
            guard let http = response as? HTTPURLResponse else {
                throw HermesError.transport("No HTTP response")
            }
            switch http.statusCode {
            case 200...299: return data
            case 401, 403: throw HermesError.unauthorized
            default:
                let msg = String(data: data, encoding: .utf8) ?? ""
                throw HermesError.server(http.statusCode, msg)
            }
        } catch let e as HermesError {
            throw e
        } catch {
            throw HermesError.transport(error.localizedDescription)
        }
    }

    private func rawGet(_ path: String) async throws -> Data {
        try await request(method: "GET", path: path, body: nil)
    }

    private func get<T: Decodable>(_ path: String, decode: T.Type) async throws -> T {
        let data = try await request(method: "GET", path: path, body: nil)
        return try decodeOrThrow(data)
    }

    private func post<T: Decodable>(_ path: String, json: [String: Any], decode: T.Type) async throws -> T {
        let body = try JSONSerialization.data(withJSONObject: json)
        let data = try await request(method: "POST", path: path, body: body)
        return try decodeOrThrow(data)
    }

    private func decodeOrThrow<T: Decodable>(_ data: Data) throws -> T {
        do { return try JSONDecoder().decode(T.self, from: data) }
        catch { throw HermesError.decoding(String(describing: error)) }
    }
}
