import Foundation

/// A JSON value exchanged with the Hermes HTTP and JSON-RPC APIs.
public indirect enum JSON: Codable, Equatable, Sendable {
    case object([String: JSON])
    case array([JSON])
    case string(String)
    case number(Double)
    case bool(Bool)
    case null

    public subscript(key: String) -> JSON {
        guard case let .object(object) = self else { return .null }
        return object[key] ?? .null
    }

    public var string: String? {
        guard case let .string(value) = self else { return nil }
        return value
    }

    public var bool: Bool? {
        guard case let .bool(value) = self else { return nil }
        return value
    }

    public var array: [JSON]? {
        guard case let .array(value) = self else { return nil }
        return value
    }

    public var object: [String: JSON]? {
        guard case let .object(value) = self else { return nil }
        return value
    }

    public var int: Int? {
        guard case let .number(value) = self, value.isFinite else { return nil }
        return Int(exactly: value)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSON].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSON].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .object(value): try container.encode(value)
        case let .array(value): try container.encode(value)
        case let .string(value): try container.encode(value)
        case let .number(value): try container.encode(value)
        case let .bool(value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}
