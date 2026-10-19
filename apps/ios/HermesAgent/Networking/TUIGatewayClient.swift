import Foundation

/// Thin JSON-RPC 2.0 client for the Hermes TUI Gateway over WebSocket (`/api/ws`).
/// Used for slash-command dispatch (`command.dispatch`) and other control-plane
/// calls that aren't part of the OpenAI-compatible REST surface.
///
/// Connection is lazy: we only dial the socket when a slash command is actually
/// dispatched, so the common chat path stays REST/SSE-only.
final class TUIGatewayClient: NSObject {
    private let connection: Connection
    private var task: URLSessionWebSocketTask?
    private var session: URLSession!
    private var nextId = 1
    private var pending: [Int: CheckedContinuation<[String: Any], Error>] = [:]
    private let lock = NSLock()

    init(connection: Connection) {
        self.connection = connection
        super.init()
        self.session = URLSession(configuration: .default)
    }

    /// Dispatch a slash command such as `/model`, `/skills`, `/new`.
    /// Returns the JSON-RPC `result` payload.
    @discardableResult
    func dispatch(command: String, sessionId: String?) async throws -> [String: Any] {
        try ensureConnected()
        var params: [String: Any] = ["command": command]
        if let sessionId { params["session_id"] = sessionId }
        return try await call(method: "command.dispatch", params: params)
    }

    func disconnect() {
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
    }

    // MARK: - Internals

    private func ensureConnected() throws {
        if task != nil { return }
        guard let url = connection.webSocketURL else { throw HermesError.badURL }
        var req = URLRequest(url: url)
        if !connection.apiKey.isEmpty {
            req.setValue("Bearer \(connection.apiKey)", forHTTPHeaderField: "Authorization")
        }
        let t = session.webSocketTask(with: req)
        task = t
        t.resume()
        receiveLoop()
    }

    private func call(method: String, params: [String: Any]) async throws -> [String: Any] {
        let id: Int = {
            lock.lock(); defer { lock.unlock() }
            let value = nextId; nextId += 1; return value
        }()
        let payload: [String: Any] = [
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        guard let text = String(data: data, encoding: .utf8) else {
            throw HermesError.transport("Failed to encode request")
        }

        return try await withCheckedThrowingContinuation { cont in
            lock.lock(); pending[id] = cont; lock.unlock()
            task?.send(.string(text)) { [weak self] error in
                if let error {
                    self?.resume(id: id, with: .failure(HermesError.transport(error.localizedDescription)))
                }
            }
        }
    }

    private func receiveLoop() {
        task?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let error):
                self.failAll(error: HermesError.transport(error.localizedDescription))
            case .success(let message):
                if case let .string(text) = message,
                   let data = text.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    self.handle(json)
                }
                self.receiveLoop()
            }
        }
    }

    private func handle(_ json: [String: Any]) {
        // Responses carry an `id`; notifications/events do not.
        guard let id = json["id"] as? Int else { return }
        if let error = json["error"] as? [String: Any] {
            let msg = error["message"] as? String ?? "RPC error"
            resume(id: id, with: .failure(HermesError.transport(msg)))
        } else {
            let result = json["result"] as? [String: Any] ?? [:]
            resume(id: id, with: .success(result))
        }
    }

    private func resume(id: Int, with result: Result<[String: Any], Error>) {
        lock.lock()
        let cont = pending.removeValue(forKey: id)
        lock.unlock()
        switch result {
        case .success(let value): cont?.resume(returning: value)
        case .failure(let error): cont?.resume(throwing: error)
        }
    }

    private func failAll(error: Error) {
        lock.lock()
        let conts = pending.values
        pending.removeAll()
        lock.unlock()
        conts.forEach { $0.resume(throwing: error) }
    }
}
