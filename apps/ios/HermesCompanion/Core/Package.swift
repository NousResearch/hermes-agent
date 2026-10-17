// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "HermesCore",
    platforms: [.macOS(.v13), .iOS(.v17)],
    products: [
        .library(name: "HermesCore", targets: ["HermesCore"]),
    ],
    targets: [
        .target(name: "HermesCore"),
        .testTarget(name: "HermesCoreTests", dependencies: ["HermesCore"]),
    ]
)
