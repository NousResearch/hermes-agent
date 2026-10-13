import Foundation

/// An entry from `GET /v1/models` (OpenAI-compatible list shape).
struct ModelInfo: Identifiable, Codable, Hashable {
    let id: String
    var object: String?
    var ownedBy: String?

    enum CodingKeys: String, CodingKey {
        case id, object
        case ownedBy = "owned_by"
    }
}

struct ModelListResponse: Codable {
    let object: String?
    let data: [ModelInfo]
}

/// A skill from `GET /v1/skills`. The server returns loosely-typed entries, so
/// we decode opportunistically.
struct SkillInfo: Identifiable, Codable, Hashable {
    var name: String
    var description: String?

    var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name, description
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = (try? c.decode(String.self, forKey: .name)) ?? "skill"
        description = try? c.decode(String.self, forKey: .description)
    }
}
