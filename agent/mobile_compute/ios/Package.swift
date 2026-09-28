// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MobileCompute",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .executable(name: "MobileCompute", targets: ["MobileCompute"]),
    ],
    dependencies: [
        // SwiftNIO for async networking
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.65.0"),
        // SwiftNIO HTTP for HTTP/1.1 server
        .package(url: "https://github.com/apple/swift-nio-http.git", from: "1.1.0"),
        // SwiftNIO SSL for TLS support
        .package(url: "https://github.com/apple/swift-nio-ssl.git", from: "2.25.0"),
    ],
    targets: [
        .executableTarget(
            name: "MobileCompute",
            dependencies: [
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio-http"),
                .product(name: "NIOSSL", package: "swift-nio-ssl"),
            ],
            swiftSettings: [
                .define("DEBUG", .when(configuration: .debug)),
            ]
        ),
        .testTarget(
            name: "MobileComputeTests",
            dependencies: ["MobileCompute"]
        ),
    ]
)