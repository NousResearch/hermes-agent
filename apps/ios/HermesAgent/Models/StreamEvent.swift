import Foundation

/// Decoded representation of one Server-Sent Event from
/// `POST /api/sessions/{id}/chat/stream`.
///
/// The Hermes stream emits named events:
///   run.started, message.started, assistant.delta, tool.started,
///   tool.completed, tool.progress, assistant.completed, run.completed,
///   error, done
enum StreamEvent {
    case runStarted(runId: String?)
    case messageStarted(messageId: String?)
    case assistantDelta(messageId: String?, text: String)
    case toolStarted(id: String, toolName: String, preview: String?)
    case toolCompleted(id: String, toolName: String, preview: String?)
    case toolFailed(id: String, toolName: String, preview: String?)
    case reasoningDelta(text: String)
    case assistantCompleted(messageId: String?, content: String?)
    case runCompleted(usage: Usage?)
    case approvalRequest(runId: String?, choices: [String])
    case error(String)
    case done
    case unknown(name: String)

    struct Usage {
        var inputTokens: Int?
        var outputTokens: Int?
    }

    /// Parse a raw SSE (event name + JSON data payload) into a `StreamEvent`.
    static func parse(event: String, data: String) -> StreamEvent {
        let json = data.data(using: .utf8)
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]

        func str(_ key: String) -> String? { json[key] as? String }
        func int(_ key: String) -> Int? {
            if let i = json[key] as? Int { return i }
            if let d = json[key] as? Double { return Int(d) }
            return nil
        }

        switch event {
        case "run.started":
            return .runStarted(runId: str("run_id"))
        case "message.started":
            return .messageStarted(messageId: str("message_id"))
        case "assistant.delta", "message.delta":
            return .assistantDelta(messageId: str("message_id"), text: str("delta") ?? "")
        case "tool.started":
            return .toolStarted(id: str("tool_call_id") ?? str("message_id") ?? UUID().uuidString,
                                toolName: str("tool_name") ?? "tool",
                                preview: str("preview"))
        case "tool.completed":
            return .toolCompleted(id: str("tool_call_id") ?? str("message_id") ?? UUID().uuidString,
                                  toolName: str("tool_name") ?? "tool",
                                  preview: str("preview"))
        case "tool.failed":
            return .toolFailed(id: str("tool_call_id") ?? str("message_id") ?? UUID().uuidString,
                               toolName: str("tool_name") ?? "tool",
                               preview: str("preview"))
        case "tool.progress", "reasoning.available":
            // Hermes routes "thinking" through tool.progress with tool_name == "_thinking".
            if (str("tool_name") ?? "") == "_thinking" {
                return .reasoningDelta(text: str("delta") ?? str("preview") ?? "")
            }
            return .toolStarted(id: str("tool_call_id") ?? UUID().uuidString,
                                toolName: str("tool_name") ?? "tool",
                                preview: str("preview"))
        case "assistant.completed":
            return .assistantCompleted(messageId: str("message_id"), content: str("content"))
        case "run.completed":
            var usage: Usage?
            if let u = json["usage"] as? [String: Any] {
                usage = Usage(inputTokens: u["input_tokens"] as? Int,
                              outputTokens: u["output_tokens"] as? Int)
            }
            return .runCompleted(usage: usage)
        case "approval.request":
            let choices = (json["choices"] as? [String]) ?? ["once", "session", "always", "deny"]
            return .approvalRequest(runId: str("run_id"), choices: choices)
        case "error", "run.failed":
            return .error(str("message") ?? str("error") ?? "Unknown error")
        case "done":
            return .done
        case "message", "":
            // Unnamed `data:` line; OpenAI-compatible streams send `[DONE]` here.
            return data.trimmingCharacters(in: .whitespaces) == "[DONE]" ? .done : .unknown(name: "message")
        default:
            return .unknown(name: event)
        }
    }
}
