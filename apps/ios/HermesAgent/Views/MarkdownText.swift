import SwiftUI

/// Lightweight markdown renderer. Splits content into fenced code blocks (which
/// get a monospaced, horizontally-scrollable card) and inline-markdown prose
/// (rendered via SwiftUI's built-in `AttributedString(markdown:)`).
///
/// Deliberately dependency-free; for richer rendering a package like
/// swift-markdown-ui can be dropped in later behind the same call site.
struct MarkdownText: View {
    let raw: String

    init(_ raw: String) { self.raw = raw }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(segments.enumerated()), id: \.offset) { _, segment in
                switch segment {
                case .prose(let text):
                    Text(inlineMarkdown(text))
                        .foregroundStyle(Theme.textPrimary)
                        .fixedSize(horizontal: false, vertical: true)
                case .code(let language, let code):
                    CodeBlock(language: language, code: code)
                }
            }
        }
    }

    private enum Segment {
        case prose(String)
        case code(language: String?, code: String)
    }

    private var segments: [Segment] {
        var result: [Segment] = []
        var inCode = false
        var language: String?
        var buffer: [String] = []

        func flush(asCode: Bool) {
            let joined = buffer.joined(separator: "\n")
            if asCode {
                result.append(.code(language: language, code: joined))
            } else if !joined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                result.append(.prose(joined))
            }
            buffer.removeAll()
        }

        for line in raw.components(separatedBy: "\n") {
            if line.hasPrefix("```") {
                flush(asCode: inCode)
                if !inCode { language = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces) }
                inCode.toggle()
            } else {
                buffer.append(line)
            }
        }
        flush(asCode: inCode)
        return result
    }

    private func inlineMarkdown(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace
        )
        if let attributed = try? AttributedString(markdown: text, options: options) {
            return attributed
        }
        return AttributedString(text)
    }
}

private struct CodeBlock: View {
    let language: String?
    let code: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let language, !language.isEmpty {
                Text(language)
                    .font(.caption2.monospaced())
                    .foregroundStyle(Theme.textFaint)
                    .padding(.horizontal, 12).padding(.top, 8)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(.footnote, design: .monospaced))
                    .foregroundStyle(Theme.textPrimary)
                    .padding(12)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.stroke, lineWidth: 1))
    }
}
