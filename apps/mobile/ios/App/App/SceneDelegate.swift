import UIKit

// UIScene lifecycle (mandatory as of iOS 26/27). The root view controller
// (Capacitor's CAPBridgeViewController) is instantiated by the "Main"
// storyboard, which is declared as UISceneStoryboardFile in Info.plist's
// scene manifest. An empty UIWindowSceneSessionRole delegate therefore
// suffices.
class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?
}
