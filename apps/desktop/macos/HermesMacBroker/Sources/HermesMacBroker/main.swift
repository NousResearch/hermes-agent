import AppKit
import ApplicationServices
import AVFoundation
import CoreGraphics
import CryptoKit
import Darwin
import Foundation
import ServiceManagement
import UserNotifications

let brokerBundleIdentifier = "com.nousresearch.hermes.macbroker"
let brokerVersion = "0.2.0"
let maxRequestBytes = 1_048_576

func jsonData(_ value: Any) -> Data? {
    guard JSONSerialization.isValidJSONObject(value) else { return nil }
    return try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
}

func jsonString(_ value: Any) -> String? {
    guard let data = jsonData(value) else { return nil }
    return String(data: data, encoding: .utf8)
}

func jsonPrint(_ value: Any) {
    if let text = jsonString(value) {
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
    // UNUserNotificationCenter can raise an Objective-C exception for an
    // unbundled swiftc-built helper. Keep status probing safe for integration
    // tests and direct CLI diagnostics; the packaged LoginItem has a bundle id.
    guard Bundle.main.bundleIdentifier != nil else { return "unavailable" }
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

@available(macOS 13.0, *)
func serviceStatusName(_ status: SMAppService.Status) -> String {
    switch status {
    case .enabled:
        return "enabled"
    case .notRegistered:
        return "notRegistered"
    case .notFound:
        return "notFound"
    case .requiresApproval:
        return "requiresApproval"
    @unknown default:
        return "unknown"
    }
}

func serviceStatusPayload() -> [String: Any] {
    if #available(macOS 13.0, *) {
        let service = SMAppService.loginItem(identifier: brokerBundleIdentifier)
        return ["ok": true, "bundleId": brokerBundleIdentifier, "status": serviceStatusName(service.status)]
    }
    return ["ok": false, "error": "SMAppService requires macOS 13+"]
}

func registerLoginItem() -> [String: Any] {
    if #available(macOS 13.0, *) {
        do {
            try SMAppService.loginItem(identifier: brokerBundleIdentifier).register()
            return serviceStatusPayload()
        } catch {
            return ["ok": false, "error": error.localizedDescription, "bundleId": brokerBundleIdentifier]
        }
    }
    return ["ok": false, "error": "SMAppService requires macOS 13+"]
}

func unregisterLoginItem() -> [String: Any] {
    if #available(macOS 13.0, *) {
        do {
            try SMAppService.loginItem(identifier: brokerBundleIdentifier).unregister()
            return serviceStatusPayload()
        } catch {
            return ["ok": false, "error": error.localizedDescription, "bundleId": brokerBundleIdentifier]
        }
    }
    return ["ok": false, "error": "SMAppService requires macOS 13+"]
}

func jsonEscapedString(_ string: String) -> String {
    if let data = try? JSONSerialization.data(withJSONObject: [string], options: []),
       let text = String(data: data, encoding: .utf8),
       text.count >= 2 {
        return String(text.dropFirst().dropLast())
    }
    return "\"\(string.replacingOccurrences(of: "\"", with: "\\\""))\""
}

func canonicalJson(_ value: Any) -> String {
    guard JSONSerialization.isValidJSONObject(value),
          let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes]),
          let text = String(data: data, encoding: .utf8) else {
        return "null"
    }
    return text
}

func hmacHex(payload: String, token: String) -> String {
    let key = SymmetricKey(data: Data(token.utf8))
    let code = HMAC<SHA256>.authenticationCode(for: Data(payload.utf8), using: key)
    return code.map { String(format: "%02x", $0) }.joined()
}

func timingSafeEqual(_ left: String, _ right: String) -> Bool {
    let l = Array(left.utf8)
    let r = Array(right.utf8)
    var diff = UInt8(l.count ^ r.count)
    for index in 0..<max(l.count, r.count) {
        diff |= (index < l.count ? l[index] : 0) ^ (index < r.count ? r[index] : 0)
    }
    return diff == 0
}

let brokerProtocolVersion = 1
let brokerClockSkewMs = 5_000.0
let brokerNonceCacheLimit = 1024
var brokerSeenNonces: [String: Double] = [:]

func rememberBrokerNonce(_ nonce: String, expiresAt: Double, nowMs: Double) -> Bool {
    for (cachedNonce, cachedExpiry) in brokerSeenNonces where cachedExpiry + brokerClockSkewMs < nowMs {
        brokerSeenNonces.removeValue(forKey: cachedNonce)
    }
    if brokerSeenNonces[nonce] != nil {
        return false
    }
    if brokerSeenNonces.count >= brokerNonceCacheLimit,
       let oldest = brokerSeenNonces.min(by: { $0.value < $1.value })?.key {
        brokerSeenNonces.removeValue(forKey: oldest)
    }
    brokerSeenNonces[nonce] = expiresAt
    return true
}

func verifyBrokerEnvelope(_ request: [String: Any], token: String) -> (ok: Bool, method: String?, params: [String: Any], error: String?) {
    guard token.count >= 32 else {
        return (false, nil, [:], "broker token must be at least 32 characters")
    }
    guard let version = request["version"] as? NSNumber, version.intValue == brokerProtocolVersion else {
        return (false, nil, [:], "unsupported broker protocol version")
    }
    guard let method = request["method"] as? String else {
        return (false, nil, [:], "missing method")
    }
    guard method == "permission.status" else {
        return (false, nil, [:], "unsupported broker method: \(method)")
    }
    guard let id = request["id"] as? String, !id.isEmpty else {
        return (false, nil, [:], "missing id")
    }
    guard let nonce = request["nonce"] as? String, !nonce.isEmpty else {
        return (false, nil, [:], "missing nonce")
    }
    guard let signature = request["signature"] as? String, !signature.isEmpty else {
        return (false, nil, [:], "missing signature")
    }
    guard let issuedAt = request["issuedAt"] as? NSNumber,
          let expiresAt = request["expiresAt"] as? NSNumber else {
        return (false, nil, [:], "missing broker timestamps")
    }
    guard expiresAt.doubleValue > issuedAt.doubleValue else {
        return (false, nil, [:], "request expiry must be after issue time")
    }
    let nowMs = Date().timeIntervalSince1970 * 1000
    if issuedAt.doubleValue - brokerClockSkewMs > nowMs {
        return (false, nil, [:], "request issuedAt is in the future")
    }
    if expiresAt.doubleValue + brokerClockSkewMs < nowMs {
        return (false, nil, [:], "request expired")
    }
    var unsigned = request
    unsigned.removeValue(forKey: "signature")
    let expected = hmacHex(payload: canonicalJson(unsigned), token: token)
    guard timingSafeEqual(expected, signature) else {
        return (false, nil, [:], "signature mismatch")
    }
    guard rememberBrokerNonce(nonce, expiresAt: expiresAt.doubleValue, nowMs: nowMs) else {
        return (false, nil, [:], "request nonce replayed")
    }
    return (true, method, request["params"] as? [String: Any] ?? [:], nil)
}

func brokerResponse(_ request: [String: Any], token: String) -> [String: Any] {
    let verification = verifyBrokerEnvelope(request, token: token)
    guard verification.ok, let method = verification.method else {
        return ["ok": false, "id": request["id"] ?? NSNull(), "error": verification.error ?? "invalid request"]
    }
    switch method {
    case "permission.status":
        var payload = permissionStatusPayload()
        payload["id"] = request["id"] ?? NSNull()
        payload["method"] = method
        return payload
    default:
        return ["ok": false, "id": request["id"] ?? NSNull(), "error": "unsupported broker method: \(method)"]
    }
}

func writeJsonLine(_ value: Any, to fd: Int32) {
    let text = (jsonString(value) ?? "{\"ok\":false,\"error\":\"invalid-json\"}") + "\n"
    let bytes = Array(text.utf8)
    bytes.withUnsafeBytes { raw in
        _ = Darwin.write(fd, raw.baseAddress, raw.count)
    }
}

func readRequestLine(from fd: Int32) -> Data? {
    var result = Data()
    var buffer = [UInt8](repeating: 0, count: 4096)
    while result.count < maxRequestBytes {
        let count = buffer.withUnsafeMutableBytes { raw in
            Darwin.read(fd, raw.baseAddress, raw.count)
        }
        if count <= 0 { break }
        if let newline = buffer[..<count].firstIndex(of: UInt8(ascii: "\n")) {
            result.append(contentsOf: buffer[..<newline])
            break
        }
        result.append(contentsOf: buffer[..<count])
    }
    return result.isEmpty ? nil : result
}

func handleClient(fd: Int32, token: String) {
    defer { Darwin.close(fd) }
    guard let data = readRequestLine(from: fd) else {
        writeJsonLine(["ok": false, "error": "empty request"], to: fd)
        return
    }
    do {
        guard let request = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            writeJsonLine(["ok": false, "error": "request must be a JSON object"], to: fd)
            return
        }
        writeJsonLine(brokerResponse(request, token: token), to: fd)
    } catch {
        writeJsonLine(["ok": false, "error": "invalid JSON: \(error.localizedDescription)"], to: fd)
    }
}

func serve(socketPath: String, token: String) -> Never {
    let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else {
        jsonPrint(["ok": false, "error": "socket failed: \(String(cString: strerror(errno)))"])
        exit(1)
    }
    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let pathBytes = Array(socketPath.utf8CString)
    let maxPath = MemoryLayout.size(ofValue: addr.sun_path)
    guard pathBytes.count <= maxPath else {
        jsonPrint(["ok": false, "error": "socket path too long"])
        exit(1)
    }
    _ = socketPath.withCString { path in unlink(path) }
    withUnsafeMutablePointer(to: &addr.sun_path) { pointer in
        pointer.withMemoryRebound(to: CChar.self, capacity: maxPath) { chars in
            for index in 0..<pathBytes.count {
                chars[index] = pathBytes[index]
            }
        }
    }
    let len = socklen_t(MemoryLayout<sockaddr_un>.offset(of: \.sun_path)! + pathBytes.count)
    let bindResult = withUnsafePointer(to: &addr) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
            Darwin.bind(fd, sockaddrPointer, len)
        }
    }
    guard bindResult == 0 else {
        jsonPrint(["ok": false, "error": "bind failed: \(String(cString: strerror(errno)))", "socket": socketPath])
        exit(1)
    }
    chmod(socketPath, 0o600)
    guard Darwin.listen(fd, 16) == 0 else {
        jsonPrint(["ok": false, "error": "listen failed: \(String(cString: strerror(errno)))"])
        exit(1)
    }
    jsonPrint(["ok": true, "event": "listening", "socket": socketPath, "bundleId": brokerBundleIdentifier, "version": brokerVersion])
    fflush(stdout)
    while true {
        let client = Darwin.accept(fd, nil, nil)
        if client >= 0 {
            handleClient(fd: client, token: token)
        }
    }
}

func argValue(_ args: [String], _ name: String) -> String? {
    guard let index = args.firstIndex(of: name), index + 1 < args.count else { return nil }
    return args[index + 1]
}

func usage() {
    jsonPrint([
        "ok": false,
        "error": "usage",
        "commands": [
            "--status-json",
            "--service-status",
            "--register-login-item",
            "--unregister-login-item",
            "--open-settings <accessibility|screenCapture|microphone|notifications|automation>",
            "--notify <title> <body>",
            "--serve --socket <path> --token <token>"
        ]
    ])
}

let args = Array(CommandLine.arguments.dropFirst())

switch args.first {
case "--status-json":
    jsonPrint(permissionStatusPayload())
case "--service-status":
    jsonPrint(serviceStatusPayload())
case "--register-login-item":
    jsonPrint(registerLoginItem())
case "--unregister-login-item":
    jsonPrint(unregisterLoginItem())
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
case "--serve":
    guard let socketPath = argValue(args, "--socket"), let token = argValue(args, "--token") else {
        usage()
        exit(2)
    }
    serve(socketPath: socketPath, token: token)
case "--help", "-h", nil:
    usage()
default:
    usage()
    exit(2)
}
