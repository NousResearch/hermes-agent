// swift-tools-version: 5.9
// MobileCompute - SwiftNIO HTTP server for the Hermes Mobile Compute node.
//
// BUILD SYSTEM OF RECORD
// ----------------------
// The shipping build system is the Xcode application target:
//
//   agent/mobile_compute/ios/MobileCompute.xcodeproj  (target: MobileCompute)
//
// That target declares its own SwiftNIO package reference (swift-nio only)
// and compiles Sources/MobileCompute/*.swift against UIKit with bundle id
// com.hermes.mobilecompute and INFOPLIST_FILE = Info.plist.
//
// DEPENDENCY NOTE
// ---------------
// NIOHTTP1 and NIOHTTP1Server ship INSIDE the swift-nio package. There is
// exactly one SwiftNIO package reference here.
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
        // Single SwiftNIO package: NIOCore, NIOPosix and NIOHTTP1 all come
        // from one package.
        //
        // Pinned to the 2.86.2 line on purpose. swift-nio 2.87.0+ requires
        // swift-tools-version 6.0, and 2.98.0+ requires 6.1, so a
        // `from:`/upToNextMajor range would let SPM pick a manifest the
        // Xcode 15.4 toolchain (Swift 5.10) cannot parse. 2.86.2 is the last
        // release declaring swift-tools-version 5.10, and it is a patch of
        // 2.86.0 so all known fixes in the line are included.
        .package(url: "https://github.com/apple/swift-nio.git", "2.86.0"..<"2.87.0"),
    ],
    targets: [
        .executableTarget(
            name: "MobileCompute",
            dependencies: [
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            swiftSettings: [
                .define("DEBUG", .when(configuration: .debug)),
            ]
        ),
    ]
)
