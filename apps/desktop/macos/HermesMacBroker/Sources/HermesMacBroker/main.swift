import AppKit
import ApplicationServices
import AVFoundation
import CoreGraphics
import Foundation
import UserNotifications

let brokerBundleIdentifier = "com.nousresearch.hermes.macbroker"
let brokerVersion = "0.1.0"

func jsonPrint(_ value: Any) {
    if JSONSerialization.isValidJSONObject(value),
       let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
       let text = String(data: data, encoding: .utf8) {
        print(text)
    } else {
        print("{\"ok\":false,\"error\":\"invalid-json\"}")
    }
}

func microphoneStatus() -> String {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return "authorized"
    case .denied:
        return "denied"
    case .restricted:
        return "restricted"
    case .notDetermined:
        return "notDetermined"
    @unknown default:
        return "unknown"
    }
}

func notificationStatus() -> String {
    let semaphore = DispatchSemaphore(value: 0)
    var status = "unknown"
    UNUserNotificationCenter.current().getNotificationSettings { settings in
        switch settings.authorizationStatus {
        case .authorized:
            status = "authorized"
        case .denied:
            status = "denied"
        case .notDetermined:
            status = "notDetermined"
        case .provisional:
            status = "provisional"
        case .ephemeral:
            status = "ephemeral"
        @unknown default:
            status = "unknown"
        }
        semaphore.signal()
    }
    _ = semaphore.wait(timeout: .now() + 2.0)
    return status
}

func screenCaptureStatus() -> String {
    if #available(macOS 10.15, *) {
        return CGPreflightScreenCaptureAccess() ? "authorized" : "notAuthorized"
    }
    return "unsupported"
}

func accessibilityStatus() -> String {
    AXIsProcessTrusted() ? "authorized" : "notAuthorized"
}

func permissionStatusPayload() -> [String: Any] {
    [
        "ok": true,
        "broker": [
            "bundleId": brokerBundleIdentifier,
            "version": brokerVersion,
            "pid": ProcessInfo.processInfo.processIdentifier,
            "executable": CommandLine.arguments.first ?? "HermesMacBroker"
        ],
        "permissions": [
            "accessibility": accessibilityStatus(),
            "screenCapture": screenCaptureStatus(),
            "microphone": microphoneStatus(),
            "notifications": notificationStatus(),
            "appleEvents": "notPreflightable"
        ]
    ]
}

func openSettings(_ pane: String) -> Bool {
    let anchors: [String: String] = [
        "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        "screenCapture": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
        "screen": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
        "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
        "notifications": "x-apple.systempreferences:com.apple.Notifications-Settings.extension",
        "automation": "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation",
        "appleEvents": "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation"
    ]
    guard let raw = anchors[pane], let url = URL(string: raw) else {
        return false
    }
    return NSWorkspace.shared.open(url)
}

func sendNotification(title: String, body: String) {
    let center = UNUserNotificationCenter.current()
    center.requestAuthorization(options: [.alert, .sound]) { _, _ in
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        let request = UNNotificationRequest(identifier: "hermes-macbroker-\(UUID().uuidString)", content: content, trigger: nil)
        center.add(request) { error in
            if let error = error {
                jsonPrint(["ok": false, "error": error.localizedDescription])
            } else {
                jsonPrint(["ok": true])
            }
            CFRunLoopStop(CFRunLoopGetMain())
        }
    }
    CFRunLoopRun()
}

func usage() {
    jsonPrint([
        "ok": false,
        "error": "usage",
        "commands": [
            "--status-json",
            "--open-settings <accessibility|screenCapture|microphone|notifications|automation>",
            "--notify <title> <body>"
        ]
    ])
}

let args = Array(CommandLine.arguments.dropFirst())

switch args.first {
case "--status-json":
    jsonPrint(permissionStatusPayload())
case "--open-settings":
    guard args.count >= 2 else {
        usage()
        exit(2)
    }
    let ok = openSettings(args[1])
    jsonPrint(["ok": ok, "pane": args[1]])
case "--notify":
    guard args.count >= 3 else {
        usage()
        exit(2)
    }
    sendNotification(title: args[1], body: args[2])
case "--help", "-h", nil:
    usage()
default:
    usage()
    exit(2)
}
