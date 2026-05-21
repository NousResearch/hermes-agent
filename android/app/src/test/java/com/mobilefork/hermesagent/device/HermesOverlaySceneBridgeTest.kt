package com.mobilefork.hermesagent.device

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class HermesOverlaySceneBridgeTest {
    @Test
    fun percentWidthClampsToLandscapeSafeArea() {
        val payload = HermesOverlaySceneBridge.payloadFromArguments(
            JSONObject()
                .put("scene_text", "Landscape overlay")
                .put("width", "94%"),
        )

        val layout = HermesOverlaySceneBridge.resolveLayoutMetrics(
            payload = payload,
            density = 3f,
            screenWidthPx = 2400,
            screenHeightPx = 1080,
            safeInsetLeftPx = 120,
            safeInsetRightPx = 120,
        )

        assertEquals("fraction", layout.widthMode)
        assertEquals("landscape", layout.orientation)
        assertEquals(2160, layout.usableWidthPx)
        assertEquals(1555, layout.availableWidthPx)
        assertEquals(layout.availableWidthPx, layout.resolvedWidthPx)
        assertEquals(6, layout.textMaxLines)
        assertTrue(layout.toString(), layout.resolvedWidthPx <= layout.usableWidthPx)
    }

    @Test
    fun percentWidthKeepsNarrowPortraitInsideMargins() {
        val payload = HermesOverlaySceneBridge.payloadFromArguments(
            JSONObject()
                .put("scene_text", "Narrow portrait overlay")
                .put("width", "98%"),
        )

        val layout = HermesOverlaySceneBridge.resolveLayoutMetrics(
            payload = payload,
            density = 2f,
            screenWidthPx = 640,
            screenHeightPx = 1400,
            safeInsetTopPx = 80,
            safeInsetBottomPx = 120,
        )

        assertEquals("portrait", layout.orientation)
        assertEquals(16, layout.edgeMarginPx)
        assertEquals(608, layout.availableWidthPx)
        assertEquals(layout.availableWidthPx, layout.resolvedWidthPx)
        assertEquals(6, layout.textMaxLines)
        assertTrue(layout.toString(), layout.resolvedWidthPx <= layout.usableWidthPx - (layout.edgeMarginPx * 2))
    }

    @Test
    fun oversizedPixelWidthClampsForInsetHeavyDisplays() {
        val payload = HermesOverlaySceneBridge.payloadFromArguments(
            JSONObject()
                .put("scene_text", "Inset heavy display overlay")
                .put("width_px", 4000),
        )

        val layout = HermesOverlaySceneBridge.resolveLayoutMetrics(
            payload = payload,
            density = 2.5f,
            screenWidthPx = 1800,
            screenHeightPx = 2200,
            safeInsetLeftPx = 240,
            safeInsetTopPx = 100,
            safeInsetRightPx = 240,
            safeInsetBottomPx = 180,
        )

        assertEquals("px", layout.widthMode)
        assertEquals("portrait", layout.orientation)
        assertEquals(1320, layout.usableWidthPx)
        assertEquals(1214, layout.availableWidthPx)
        assertEquals(layout.availableWidthPx, layout.resolvedWidthPx)
        assertEquals(12, layout.textMaxLines)
        assertTrue(layout.toJson().toString(), layout.toJson().has("screen_aspect_ratio"))
    }
}
