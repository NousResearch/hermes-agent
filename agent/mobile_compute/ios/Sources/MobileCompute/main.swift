// MobileCompute.swift - Entry point for the Mobile Compute Node iOS app.
//
// This is the main entry point for the Swift Mobile Compute server.
// It starts an HTTP server on the specified port that implements
// the Mobile Compute API protocol compatible with the Hermes
// MobileComputeClient.

import Foundation
import NIOCore
import NIOPosix
import NIOHTTP1
import NIOSSL

@main
struct MobileComputeMain {
    static func main() async {
        let config = MobileComputeConfig.load()
        
        guard config.enabled else {
            print("Mobile Compute is disabled. Set enabled=true in config.")
            return
        }
        
        do {
            let server = try await MobileComputeServer(config: config)
            try await server.start()
            print("Mobile Compute server listening on \(config.host):\(config.port)")
            
            // Keep running until interrupted
            while !Task.isCancelled {
                try await Task.sleep(nanoseconds: 1_000_000_000)
            }
            
            await server.stop()
        } catch {
            print("Failed to start Mobile Compute server: \(error)")
            exit(1)
        }
    }
}