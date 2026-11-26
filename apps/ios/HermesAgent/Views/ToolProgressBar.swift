import SwiftUI

/// The "Tool Execution" status bar that sits above the composer while the agent
/// is working — fed by `tool.started` / `tool.completed` / `tool.progress` SSE
/// events (the PRD's headline real-time feature).
struct ToolProgressBar: View {
    let tools: [ToolEvent]
    let status: String?
    let isStreaming: Bool

    var body: some View {
        if isStreaming || !tools.isEmpty {
            VStack(spacing: 6) {
                if let status {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                            .tint(Theme.accent)
                        Text(status)
                            .font(.caption.weight(.medium))
                            .foregroundStyle(Theme.textSecondary)
                            .lineLimit(1)
                        Spacer()
                    }
                    .padding(.horizontal, 14)
                }
                if !tools.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(tools) { ToolChip(tool: $0, compact: false) }
                        }
                        .padding(.horizontal, 14)
                    }
                }
            }
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity)
            .background(Theme.toolChip.opacity(0.5))
            .overlay(Rectangle().frame(height: 1).foregroundStyle(Theme.stroke), alignment: .top)
            .transition(.move(edge: .bottom).combined(with: .opacity))
            .animation(.easeInOut(duration: 0.2), value: tools)
        }
    }
}

/// A single tool chip with emoji, name, and a status glyph.
struct ToolChip: View {
    let tool: ToolEvent
    var compact: Bool

    var body: some View {
        HStack(spacing: 6) {
            Text(tool.symbol).font(.caption)
            Text(tool.title)
                .font(.caption.weight(.medium))
                .foregroundStyle(Theme.textPrimary)
                .lineLimit(1)
            statusGlyph
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(Theme.toolChip, in: Capsule())
        .overlay(Capsule().stroke(borderColor, lineWidth: 1))
    }

    @ViewBuilder
    private var statusGlyph: some View {
        switch tool.status {
        case .running:
            ProgressView().controlSize(.mini).tint(Theme.accent)
        case .completed:
            Image(systemName: "checkmark").font(.caption2.bold()).foregroundStyle(Theme.success)
        case .failed:
            Image(systemName: "xmark").font(.caption2.bold()).foregroundStyle(Theme.danger)
        }
    }

    private var borderColor: Color {
        switch tool.status {
        case .running: return Theme.accent.opacity(0.5)
        case .completed: return Theme.success.opacity(0.4)
        case .failed: return Theme.danger.opacity(0.5)
        }
    }
}
