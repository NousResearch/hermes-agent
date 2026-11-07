import SwiftUI

// Semantic palette translated from Hermes Desktop's theme-primary and flat surfaces.
enum HermesTheme {
    static let accent = Color(red: 0, green: 83 / 255, blue: 253 / 255)
    static let background = Color(uiColor: .systemBackground)
    static let surface = Color(uiColor: .secondarySystemBackground)
    static let hairline = Color.primary.opacity(0.08)
}
struct HermesMark: View {
    var size: CGFloat = 36
    var body: some View {
        Image("HermesMark").renderingMode(.original).resizable().scaledToFill().frame(width: size, height: size)
            .background(.white).clipShape(RoundedRectangle(cornerRadius: 7))
            .accessibilityHidden(true)
    }
}
struct QuietButton: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.frame(minWidth: 44, minHeight: 44)
            .background(configuration.isPressed ? HermesTheme.surface : .clear)
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
struct EmptyPanel: View {
    let title: String
    let detail: String
    var body: some View {
        VStack(spacing: 14) {
            HermesMark(size: 54)
            Text(title).font(.title2.weight(.semibold))
            Text(detail).font(.subheadline).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }.padding(32).frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
