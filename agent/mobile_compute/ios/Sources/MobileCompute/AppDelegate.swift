// AppDelegate.swift - UIKit lifecycle wrapper for the Mobile Compute server.
//
// The UIKit entry point is invoked from main.swift, which owns the whole
// app lifecycle that lives here. The SwiftNIO HTTP server is started once the
// application finishes launching and is shut down (channel close +
// EventLoopGroup shutdown) when the app terminates. Compute stays echo-only:
// no Core ML, no LLM.

import UIKit
import Foundation

final class AppDelegate: NSObject, UIApplicationDelegate {

    var window: UIWindow?

    private var server: MobileComputeServer?

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        // Minimal window: the app is a headless compute node, but UIKit still
        // requires a key window for the process to stay foreground-active.
        let window = UIWindow(frame: UIScreen.main.bounds)
        window.rootViewController = UIViewController()
        window.makeKeyAndVisible()
        self.window = window

        let config = MobileComputeConfig.load()

        guard config.enabled else {
            NSLog("Mobile Compute is disabled; server not started.")
            return true
        }

        let server = MobileComputeServer(config: config)
        self.server = server

        Task {
            do {
                try await server.start()
                NSLog("Mobile Compute server listening on \(config.host):\(config.port)")
            } catch {
                NSLog("Mobile Compute server failed to start: \(error)")
            }
        }

        return true
    }

    func applicationWillTerminate(_ application: UIApplication) {
        guard let server = self.server else { return }
        self.server = nil

        // iOS gives only a short termination window, so the shutdown is
        // scheduled and awaited briefly instead of blocking the main thread
        // on an async call.
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            try? await server.stop()
            semaphore.signal()
        }
        _ = semaphore.wait(timeout: .now() + 2)
    }
}
