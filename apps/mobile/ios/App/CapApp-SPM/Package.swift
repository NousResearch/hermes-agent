// swift-tools-version: 5.9
import PackageDescription

// DO NOT MODIFY THIS FILE - managed by Capacitor CLI commands
let package = Package(
    name: "CapApp-SPM",
    platforms: [.iOS(.v15)],
    products: [
        .library(
            name: "CapApp-SPM",
            targets: ["CapApp-SPM"])
    ],
    dependencies: [
        .package(url: "https://github.com/ionic-team/capacitor-swift-pm.git", exact: "8.4.1"),
        .package(name: "CapacitorCommunitySafeArea", path: "../../../../../node_modules/@capacitor-community/safe-area"),
        .package(name: "CapacitorApp", path: "../../../../../node_modules/@capacitor/app"),
        .package(name: "CapacitorBrowser", path: "../../../../../node_modules/@capacitor/browser"),
        .package(name: "CapacitorClipboard", path: "../../../../../node_modules/@capacitor/clipboard"),
        .package(name: "CapacitorFilesystem", path: "../../../../../node_modules/@capacitor/filesystem"),
        .package(name: "CapacitorHaptics", path: "../../../../../node_modules/@capacitor/haptics"),
        .package(name: "CapacitorKeyboard", path: "../../../../../node_modules/@capacitor/keyboard"),
        .package(name: "CapacitorLocalNotifications", path: "../../../../../node_modules/@capacitor/local-notifications"),
        .package(name: "CapacitorNetwork", path: "../../../../../node_modules/@capacitor/network"),
        .package(name: "CapacitorPreferences", path: "../../../../../node_modules/@capacitor/preferences"),
        .package(name: "CapacitorShare", path: "../../../../../node_modules/@capacitor/share"),
        .package(name: "CapacitorStatusBar", path: "../../../../../node_modules/@capacitor/status-bar")
    ],
    targets: [
        .target(
            name: "CapApp-SPM",
            dependencies: [
                .product(name: "Capacitor", package: "capacitor-swift-pm"),
                .product(name: "Cordova", package: "capacitor-swift-pm"),
                .product(name: "CapacitorCommunitySafeArea", package: "CapacitorCommunitySafeArea"),
                .product(name: "CapacitorApp", package: "CapacitorApp"),
                .product(name: "CapacitorBrowser", package: "CapacitorBrowser"),
                .product(name: "CapacitorClipboard", package: "CapacitorClipboard"),
                .product(name: "CapacitorFilesystem", package: "CapacitorFilesystem"),
                .product(name: "CapacitorHaptics", package: "CapacitorHaptics"),
                .product(name: "CapacitorKeyboard", package: "CapacitorKeyboard"),
                .product(name: "CapacitorLocalNotifications", package: "CapacitorLocalNotifications"),
                .product(name: "CapacitorNetwork", package: "CapacitorNetwork"),
                .product(name: "CapacitorPreferences", package: "CapacitorPreferences"),
                .product(name: "CapacitorShare", package: "CapacitorShare"),
                .product(name: "CapacitorStatusBar", package: "CapacitorStatusBar")
            ]
        )
    ]
)
