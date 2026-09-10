import Foundation

/// Streams an agent turn from `POST /api/sessions/{id}/chat/stream` and yields
/// decoded `StreamEvent`s as they arrive. Built on `URLSession.bytes(for:)`,
/// which gives us a line-by-line async byte stream without third-party deps.
struct SSEClient {
    let connection: Connection

    /// Open a streaming chat turn. The returned async stream finishes when the
    /// server emits `done` / `[DONE]` or the connection closes.
    func streamChat(sessionId: String, message: String, model: String?) -> AsyncThrowingStream<StreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard connection.isValid, let base = connection.url else {
                        throw HermesError.notConfigured
                    }
                    let path = "/api/sessions/\(sessionId)/chat/stream"
                    guard let url = URL(string: path, relativeTo: base) else { throw HermesError.badURL }

                    var req = URLRequest(url: url)
                    req.httpMethod = "POST"
                    req.timeoutInterval = 600
                    req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    if !connection.apiKey.isEmpty {
                        req.setValue("Bearer \(connection.apiKey)", forHTTPHeaderField: "Authorization")
                    }
                    // The server reads `message` (falling back to `input`); see
                    // `_session_chat_user_message` in api_server.py.
                    var body: [String: Any] = ["message": message, "stream": true]
                    if let model { body["model"] = model }
                    req.httpBody = try JSONSerialization.data(withJSONObject: body)

                    let config = URLSessionConfiguration.default
                    config.timeoutIntervalForRequest = 600
                    let session = URLSession(configuration: config)

                    let (bytes, response) = try await session.bytes(for: req)
                    if let http = response as? HTTPURLResponse {
                        if http.statusCode == 401 || http.statusCode == 403 {
                            throw HermesError.unauthorized
                        }
                        guard (200...299).contains(http.statusCode) else {
                            throw HermesError.server(http.statusCode, "stream failed")
                        }
                    }

                    var eventName = "message"
                    var dataBuffer = ""

                    for try await line in bytes.lines {
                        if Task.isCancelled { break }

                        if line.isEmpty {
                            // Blank line dispatches the accumulated event.
                            if !dataBuffer.isEmpty {
                                let event = StreamEvent.parse(event: eventName, data: dataBuffer)
                                continuation.yield(event)
                                if case .done = event { break }
                            }
                            eventName = "message"
                            dataBuffer = ""
                            continue
                        }
                        if line.hasPrefix(":") { continue } // comment / heartbeat
                        if line.hasPrefix("event:") {
                            eventName = line.dropFirst(6).trimmingCharacters(in: .whitespaces)
                        } else if line.hasPrefix("data:") {
                            let chunk = String(line.dropFirst(5).drop(while: { $0 == " " }))
                            dataBuffer += dataBuffer.isEmpty ? chunk : "\n" + chunk
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
