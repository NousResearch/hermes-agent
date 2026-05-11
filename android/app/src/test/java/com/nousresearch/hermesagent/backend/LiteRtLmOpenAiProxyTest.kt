package com.nousresearch.hermesagent.backend

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.io.path.pathString

class LiteRtLmOpenAiProxyTest {
    @Test
    fun validateModelArtifact_acceptsLiteRtLmHeader() {
        val file = tempModelFile("gemma-4-E2B-it.litertlm", "LITERTLM".toByteArray())

        assertNull(validateModelArtifact(file))
    }

    @Test
    fun validateModelArtifact_rejectsWebTaskFlatBufferBeforeEngineStart() {
        val file = tempModelFile(
            "gemma-4-E2B-it-web.task",
            byteArrayOf(0, 0, 0, 0, 'T'.code.toByte(), 'F'.code.toByte(), 'L'.code.toByte(), '3'.code.toByte()),
        )

        val error = validateModelArtifact(file).orEmpty()

        assertTrue(error, error.contains("web/browser .task FlatBuffer"))
        assertTrue(error, error.contains("download the .litertlm artifact instead"))
    }

    @Test
    fun validateModelArtifact_rejectsBrokenLiteRtLmFileWithZipHeader() {
        val file = tempModelFile("gemma-4-E4B-it.litertlm", byteArrayOf('P'.code.toByte(), 'K'.code.toByte(), 3, 4, 0, 0, 0, 0))

        val error = validateModelArtifact(file)

        assertEquals(
            "gemma-4-E4B-it.litertlm is not a valid LiteRT-LM bundle. Download the .litertlm artifact from the LiteRT-LM repo.",
            error,
        )
    }

    private fun validateModelArtifact(file: File): String? {
        val method = LiteRtLmOpenAiProxy::class.java.getDeclaredMethod(
            "validateModelArtifact",
            String::class.java,
        )
        method.isAccessible = true
        return method.invoke(LiteRtLmOpenAiProxy, file.absolutePath) as String?
    }

    private fun tempModelFile(name: String, header: ByteArray): File {
        val dir = File(createTempDirectory(prefix = "hermes-litertlm-test-").pathString)
        return File(dir, name).apply {
            writeBytes(header + ByteArray(16) { 1 })
            deleteOnExit()
            dir.deleteOnExit()
        }
    }
}
