import SwiftUI

/// One transcript row. User messages are right-aligned chips; assistant messages
/// are full-width with markdown + any tool-progress chips that ran for the turn.
struct MessageBubbleView: View {
    let message: ChatMessage
    var isStreaming: Bool = false

    var body: some View {
        switch message.role {
        case .user:
            userBubble
        case .system, .tool:
            systemNote
        case .assistant:
            assistantBubble
        }
    }

    private var userBubble: some View {
        HStack {
            Spacer(minLength: 40)
            Text(message.content)
                .foregroundStyle(Theme.textPrimary)
                .padding(.horizontal, 14).padding(.vertical, 10)
                .background(Theme.userBubble, in: RoundedRectangle(cornerRadius: Theme.bubbleCorner))
                .textSelection(.enabled)
        }
    }

    private var assistantBubble: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                HermesMark(size: 22)
                Text("Hermes")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textSecondary)
            }

            if !message.toolEvents.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(message.toolEvents) { tool in
                        ToolChip(tool: tool, compact: true)
                    }
                }
            }

            if message.content.isEmpty && isStreaming {
                TypingIndicator()
            } else {
                MarkdownText(message.content)
                    .textSelection(.enabled)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var systemNote: some View {
        HStack {
            Image(systemName: "terminal")
                .font(.caption)
            MarkdownText(message.content)
                .font(.caption)
        }
        .foregroundStyle(Theme.textSecondary)
        .padding(.horizontal, 12).padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 10))
    }
}

/// Animated three-dot indicator shown while waiting for the first delta.
struct TypingIndicator: View {
    @State private var phase = 0.0
    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3) { i in
                Circle()
                    .fill(Theme.textSecondary)
                    .frame(width: 7, height: 7)
                    .opacity(phase == Double(i) ? 1 : 0.3)
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.5).repeatForever()) { phase = 2 }
        }
    }
}

/// Collapsible reasoning / "thinking" stream shown beneath the live turn.
struct ReasoningView: View {
    let text: String
    @State private var expanded = true

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button { expanded.toggle() } label: {
                Label("Reasoning", systemImage: expanded ? "chevron.down" : "chevron.right")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(Theme.accentDim)
            }
            if expanded {
                Text(text)
                    .font(.caption.monospaced())
                    .foregroundStyle(Theme.textSecondary)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Theme.surface, in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }
}
