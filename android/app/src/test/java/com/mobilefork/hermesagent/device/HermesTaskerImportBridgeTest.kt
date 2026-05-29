package com.mobilefork.hermesagent.device

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HermesTaskerImportBridgeTest {
    @Test
    fun parseSelfHealsMarkdownFencedTaskerXml() {
        val result = HermesTaskerImportBridge.parse(
            """
            ```xml
            <TaskerData>
              <Variable>
                <nme>%MESSAGE</nme>
                <val>hello</val>
              </Variable>
            </TaskerData>
            ```
            """.trimIndent(),
        )

        assertEquals(0, result.taskCount)
        assertEquals(0, result.importedActionCount)
        assertEquals("hello", result.bundle.getJSONObject("variables").getString("MESSAGE"))
    }

    @Test
    fun parseSelfHealsProseWrappedTaskerXmlWithoutAllowingDoctype() {
        val result = HermesTaskerImportBridge.parse(
            """
            Import this safe profile:
            <TaskerData>
              <Variable>
                <nme>%DEVICE</nme>
                <val>phone</val>
              </Variable>
            </TaskerData>
            End.
            """.trimIndent(),
        )

        assertEquals("phone", result.bundle.getJSONObject("variables").getString("DEVICE"))

        val rejected = runCatching {
            HermesTaskerImportBridge.parse(
                """
                ```xml
                <!DOCTYPE TaskerData [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
                <TaskerData><Variable><nme>%DEVICE</nme><val>&xxe;</val></Variable></TaskerData>
                ```
                """.trimIndent(),
            )
        }
        assertTrue(rejected.isFailure)
        assertTrue(rejected.exceptionOrNull()?.message.orEmpty().contains("DOCTYPE"))
    }
}
