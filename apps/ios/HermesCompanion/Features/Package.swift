// swift-tools-version: 5.9
import PackageDescription
let package = Package(
    name: "HermesFeatures",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "HermesFeatures", targets: ["HermesFeatures"])],
    dependencies: [.package(path: "../Core")],
    targets: [
        .target(name: "HermesFeatures", dependencies: [.product(name: "HermesCore", package: "Core")]),
        .testTarget(name: "HermesFeaturesTests", dependencies: ["HermesFeatures"])
    ]
)
