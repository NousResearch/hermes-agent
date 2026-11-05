// main.swift - Entry point for the Mobile Compute Node iOS app.
//
// This file is Swift top-level code, so it must not declare an
// attribute-based entry point: the compiler rejects that in a module that
// contains top-level code. The UIKit entry point is expressed explicitly with
// UIApplicationMain and the whole app lifecycle lives in AppDelegate.swift.
// This is the single entry point of the target.

import UIKit

UIApplicationMain(
    CommandLine.argc,
    CommandLine.unsafeArgv,
    nil,
    NSStringFromClass(AppDelegate.self)
)
