import Cocoa
import Foundation

if CommandLine.arguments.count == 2 && CommandLine.arguments[1] == "--check-accessibility" {
    print(AXIsProcessTrusted())
    exit(0)
}

guard CommandLine.arguments.count == 6,
      let pid = Int32(CommandLine.arguments[1]),
      let x = Double(CommandLine.arguments[2]),
      let y = Double(CommandLine.arguments[3]),
      let dx = Double(CommandLine.arguments[4]),
      let dy = Double(CommandLine.arguments[5]) else {
    fatalError("usage: native-drag pid x y dx dy")
}
guard AXIsProcessTrusted() else { fatalError("Accessibility access is required") }
guard let app = NSRunningApplication(processIdentifier: pid) else { fatalError("Target app is not running") }
app.activate(options: [])
usleep(250_000)
let source = CGEventSource(stateID: .hidSystemState)
func send(_ type: CGEventType, _ x: Double, _ y: Double) {
    let event = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: CGPoint(x: x, y: y), mouseButton: .left)!
    event.setIntegerValueField(.mouseEventClickState, value: 1)
    event.post(tap: .cghidEventTap)
}
send(.mouseMoved, x, y)
usleep(100_000)
send(.leftMouseDown, x, y)
usleep(100_000)
for i in 1...20 {
    send(.leftMouseDragged, x + dx * Double(i) / 20, y + dy * Double(i) / 20)
    usleep(20_000)
}
send(.leftMouseUp, x + dx, y + dy)
usleep(200_000)
