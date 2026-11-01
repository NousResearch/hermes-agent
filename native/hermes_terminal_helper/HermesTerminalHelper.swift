// HermesTerminalHelper — sends Apple Events in-process via NSAppleScript.
//
// Why this exists: Hermes' macOS terminal raise-ladder (tools/
// terminal_tool_macos_open.py) and the profile "open terminal" endpoint
// (hermes_cli/web_routers/profiles.py) drive UI through `osascript`. macOS
// TCC attributes an Apple Event to the process that SENDS it, so every
// automation prompt reads "osascript wants to control System Events" —
// generic, unexplained, and indistinguishable from malware. Routing the
// same events through this named, signed bundle makes the prompts read
// "Hermes Terminal Helper wants to control …" with the purpose string from
// Info.plist, so the user can see what it is, what it is for, and that it
// comes from Hermes.
//
// Exit semantics mirror osascript so the raise-ladder's `if ! <query>` and
// `if <query>` branches behave identically: a successful query prints its
// value (true/false/name) and exits 0; a failed query (target not running,
// permission denied) prints nothing and exits non-zero.

import Foundation

func escapeAppleScriptString(_ value: String) -> String {
    value.replacingOccurrences(of: "\\", with: "\\\\")
         .replacingOccurrences(of: "\"", with: "\\\"")
}

func runScript(_ source: String) -> (String?, Int32) {
    guard let script = NSAppleScript(source: source) else { return (nil, 1) }
    var errorInfo: NSDictionary?
    let result = script.executeAndReturnError(&errorInfo)
    if errorInfo != nil { return (nil, 1) }
    return (result.stringValue, 0)
}

let argv = CommandLine.arguments
guard argv.count >= 2 else {
    fputs("usage: HermesTerminalHelper <activate|frontmost|unhide-all|getName|stdin> [target]\n", stderr)
    exit(2)
}

let action = argv[1]

switch action {
case "activate":
    guard argv.count >= 3 else { exit(2) }
    let app = escapeAppleScriptString(argv[2])
    exit(runScript("tell application \"\(app)\" to activate").1)

case "frontmost":
    guard argv.count >= 3 else { exit(2) }
    let app = escapeAppleScriptString(argv[2])
    let (value, code) = runScript(
        "tell application \"System Events\" to get frontmost of application process \"\(app)\""
    )
    if code != 0 { exit(code) }
    print(value ?? "false")
    exit(0)

case "unhide-all":
    exit(runScript(
        "tell application \"System Events\" to set visible of every process to true"
    ).1)

case "getName":
    let (value, code) = runScript(
        "tell application \"System Events\" to get name of first application process whose frontmost is true"
    )
    if code != 0 { exit(code) }
    print(value ?? "")
    exit(0)

case "custom":
    guard argv.count >= 3 else { exit(2) }
    let (value, code) = runScript(argv[2])
    if let value, !value.isEmpty { print(value) }
    exit(code)

case "stdin":
    let data = FileHandle.standardInput.readDataToEndOfFile()
    guard let source = String(data: data, encoding: .utf8) else { exit(2) }
    let (value, code) = runScript(source)
    if let value, !value.isEmpty { print(value) }
    exit(code)

default:
    fputs("unknown action: \(action)\n", stderr)
    exit(2)
}
