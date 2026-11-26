import Foundation

/// Mirrors a row from the Hermes `sessions` table (`hermes_state.py`).
/// Only the fields a mobile client needs are decoded; the rest are ignored so
/// the schema can evolve server-side without breaking the app.
struct HermesSession: Identifiable, Codable, Hashable {
    let id: String
    var source: String?
    var userId: String?
    var model: String?
    var title: String?
    var startedAt: Double?
    var endedAt: Double?
    var endReason: String?
    var messageCount: Int?
    var toolCallCount: Int?
    var inputTokens: Int?
    var outputTokens: Int?
    var parentSessionId: String?
    var lastActive: Double?
    var preview: String?

    enum CodingKeys: String, CodingKey {
        case id, source, model, title, preview
        case userId = "user_id"
        case startedAt = "started_at"
        case endedAt = "ended_at"
        case endReason = "end_reason"
        case messageCount = "message_count"
        case toolCallCount = "tool_call_count"
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case parentSessionId = "parent_session_id"
        case lastActive = "last_active"
    }

    /// Best-effort display name: explicit title, else the message preview, else the id.
    var displayTitle: String {
        if let title, !title.isEmpty { return title }
        if let preview, !preview.isEmpty { return preview }
        return "Session " + id.prefix(8)
    }

    var isActive: Bool { endedAt == nil }

    var lastActiveDate: Date? {
        let stamp = lastActive ?? startedAt
        return stamp.map { Date(timeIntervalSince1970: $0) }
    }
}

/// Envelope returned by `GET /api/sessions`.
struct SessionListResponse: Codable {
    let object: String?
    let data: [HermesSession]
    let limit: Int?
    let offset: Int?
    let hasMore: Bool?

    enum CodingKeys: String, CodingKey {
        case object, data, limit, offset
        case hasMore = "has_more"
    }
}

/// Envelope returned by `POST /api/sessions` and `GET /api/sessions/{id}`.
struct SessionEnvelope: Codable {
    let object: String?
    let session: HermesSession
}
