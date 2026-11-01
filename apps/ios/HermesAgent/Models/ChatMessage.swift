import Foundation

/// Mirrors a row from the Hermes `messages` table (`hermes_state.py`).
/// `id` is decoded as a string because the server may emit either the integer
/// autoincrement PK or a UUID depending on the source.
struct ChatMessage: Identifiable, Codable, Hashable {
    let id: String
    var sessionId: String?
    var role: Role
    var content: String
    var toolCallId: String?
    var toolName: String?
    var timestamp: Double?
    var tokenCount: Int?
    var finishReason: String?
    var reasoning: String?

    /// Tool-progress chips captured while this assistant turn streamed in.
    /// Not persisted server-side; populated live by the chat view model.
    var toolEvents: [ToolEvent] = []

    enum Role: String, Codable, Hashable {
        case user, assistant, system, tool
    }

    enum CodingKeys: String, CodingKey {
        case id, role, content, reasoning
        case sessionId = "session_id"
        case toolCallId = "tool_call_id"
        case toolName = "tool_name"
        case timestamp
        case tokenCount = "token_count"
        case finishReason = "finish_reason"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // id may arrive as Int (messages.id AUTOINCREMENT) or String (UUID).
        if let intId = try? c.decode(Int.self, forKey: .id) {
            id = String(intId)
        } else {
            id = (try? c.decode(String.self, forKey: .id)) ?? UUID().uuidString
        }
        sessionId = try? c.decode(String.self, forKey: .sessionId)
        role = (try? c.decode(Role.self, forKey: .role)) ?? .assistant
        content = (try? c.decode(String.self, forKey: .content)) ?? ""
        toolCallId = try? c.decode(String.self, forKey: .toolCallId)
        toolName = try? c.decode(String.self, forKey: .toolName)
        timestamp = try? c.decode(Double.self, forKey: .timestamp)
        tokenCount = try? c.decode(Int.self, forKey: .tokenCount)
        finishReason = try? c.decode(String.self, forKey: .finishReason)
        reasoning = try? c.decode(String.self, forKey: .reasoning)
    }

    /// Local-only constructor used for optimistic UI (user input, streaming draft).
    init(id: String = UUID().uuidString,
         role: Role,
         content: String,
         sessionId: String? = nil,
         timestamp: Double? = Date().timeIntervalSince1970) {
        self.id = id
        self.role = role
        self.content = content
        self.sessionId = sessionId
        self.timestamp = timestamp
    }

    var date: Date? { timestamp.map { Date(timeIntervalSince1970: $0) } }
}

/// Envelope returned by `GET /api/sessions/{id}/messages`.
struct MessageListResponse: Codable {
    let object: String?
    let sessionId: String?
    let data: [ChatMessage]

    enum CodingKeys: String, CodingKey {
        case object, data
        case sessionId = "session_id"
    }
}
