import SwiftUI

/// Hermes visual language: dark-first, minimal, high-contrast — in the spirit of
/// Linear / Raycast as called out in the UI/UX brief. A warm amber accent nods to
/// Hermes, the swift messenger.
enum Theme {
    static let accent = Color(hex: 0xE8A33D)        // Hermes amber
    static let accentDim = Color(hex: 0xB67C24)

    static let background = Color(hex: 0x0C0D10)     // near-black canvas
    static let surface = Color(hex: 0x16181D)        // cards, bubbles
    static let surfaceRaised = Color(hex: 0x1E2128)  // input bar, sheets
    static let stroke = Color(hex: 0x2A2D35)

    static let textPrimary = Color(hex: 0xF4F5F7)
    static let textSecondary = Color(hex: 0x9AA0AB)
    static let textFaint = Color(hex: 0x5C616B)

    static let userBubble = Color(hex: 0x23262E)
    static let assistantBubble = Color(hex: 0x16181D)
    static let toolChip = Color(hex: 0x1A2530)
    static let danger = Color(hex: 0xE5534B)
    static let success = Color(hex: 0x46B17A)

    static let corner: CGFloat = 14
    static let bubbleCorner: CGFloat = 18
}

extension Color {
    init(hex: UInt, alpha: Double = 1) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: alpha
        )
    }
}
