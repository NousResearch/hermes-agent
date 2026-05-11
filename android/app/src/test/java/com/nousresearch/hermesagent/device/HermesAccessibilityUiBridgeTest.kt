package com.nousresearch.hermesagent.device

import android.graphics.Bitmap
import android.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class HermesAccessibilityUiBridgeTest {
    @Test
    fun perceptualHash64IsStableBinaryAndVisual() {
        val leftDark = splitBitmap(leftColor = Color.BLACK, rightColor = Color.WHITE)
        val leftLight = splitBitmap(leftColor = Color.WHITE, rightColor = Color.BLACK)

        val firstHash = HermesAccessibilityUiBridge.perceptualHash64(leftDark)
        val repeatedHash = HermesAccessibilityUiBridge.perceptualHash64(leftDark)
        val oppositeHash = HermesAccessibilityUiBridge.perceptualHash64(leftLight)

        assertEquals(64, firstHash.length)
        assertTrue(firstHash.all { it == '0' || it == '1' })
        assertEquals(firstHash, repeatedHash)
        assertTrue(hammingDistance(firstHash, oppositeHash) >= 32)
    }

    private fun splitBitmap(leftColor: Int, rightColor: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(16, 16, Bitmap.Config.ARGB_8888)
        for (y in 0 until 16) {
            for (x in 0 until 16) {
                bitmap.setPixel(x, y, if (x < 8) leftColor else rightColor)
            }
        }
        return bitmap
    }

    private fun hammingDistance(left: String, right: String): Int {
        assertEquals(left.length, right.length)
        return left.indices.count { index -> left[index] != right[index] }
    }
}
