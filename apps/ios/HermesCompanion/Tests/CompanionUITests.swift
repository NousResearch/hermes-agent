import XCTest
final class CompanionUITests: XCTestCase {
    func testLaunchRequiresPrivateConnectionAndNoFakeConnectedStatus() {
        let app = XCUIApplication()
        app.launch()
        XCTAssertTrue(app.textFields["Tailnet URL"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.secureTextFields["Password"].exists)
        XCTAssertFalse(app.buttons["Connect to Mac"].isEnabled)
        XCTAssertFalse(app.staticTexts["Connected"].exists)
    }
    func testPreviewButtonOpensTabsWithoutLoginOrLiveControls() {
        let app = XCUIApplication()
        app.launch()
        let preview = app.buttons["preview-interface"]
        XCTAssertTrue(preview.waitForExistence(timeout: 10))
        preview.tap()
        XCTAssertTrue(app.staticTexts["Preview · not connected"].waitForExistence(timeout: 10))
        app.tabBars.buttons["Kanban"].tap()
        XCTAssertTrue(app.navigationBars["Kanban"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["Check gateway connection"].exists)
        app.tabBars.buttons["Scheduled"].tap()
        XCTAssertTrue(app.staticTexts["Morning workspace review"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["New scheduled job"].isEnabled)
        app.tabBars.buttons["Workspace"].tap()
        app.buttons["Usage charts"].tap()
        XCTAssertTrue(app.staticTexts["Sessions per day"].waitForExistence(timeout: 5))
        app.navigationBars.buttons.element(boundBy: 0).tap()
        app.buttons["New session"].tap()
        XCTAssertTrue(app.buttons["Create session"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["Create session"].isEnabled)
        app.buttons["Cancel"].tap()
        app.tabBars.buttons["Chat"].tap()
        XCTAssertTrue(app.buttons["Open conversations"].exists)
    }
}
