// swift-tools-version: 5.9
// MobileCompute - SwiftNIO HTTP server for the Hermes Mobile Compute node.
//
// BUILD SYSTEM OF RECORD
// ----------------------
// The shipping build system is the Xcode application target:
//
//   agent/mobile_compute/ios/MobileCompute.xcodeproj  (target: MobileCompute)
//
// That target declares its own SwiftNIO package references (swift-nio,
// swift-nio-http) and compiles Sources/MobileCompute/*.swift against UIKit
// with bundle id com.hermes.mobilecompute and INFOPLIST_FILE = Info.plist.
//
// This manifest is a convenience mirror for editor tooling and package
// resolution sanity checks. It is NOT what produces MobileCompute.app.
// The package is iOS-only: the entry point is a UIKit application
// (UIApplicationMain in main.swift) and cannot link on macOS, so there is no
// cross-platform SwiftPM build and no test target.

import PackageDescription

let package = Package(
    name: "MobileCompute",
    platforms: [
        .iOS(.v17),
    ],
    products: [
        .executable(name: "MobileCompute", targets: ["MobileCompute"]),
    ],
    dependencies: [
        // SwiftNIO for async networking
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.65.0"),
        // SwiftNIO HTTP for HTTP/1.1 server
        .package(url: "https://github.com/apple/swift-nio-http.git", from: "1.1.0"),
    ],
    targets: [
        .executableTarget(
            name: "MobileCompute",
            dependencies: [
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio-http"),
            ],
            swiftSettings: [
                .define("DEBUG", .when(configuration: .debug)),
            ]
        ),
    ]
)
