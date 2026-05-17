package com.nousresearch.hermesagent

import android.app.Application
import android.os.SystemClock
import android.util.Base64
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.nousresearch.hermesagent.backend.BackendKind
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.backend.OnDeviceBackendManager
import com.nousresearch.hermesagent.data.AppSettings
import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.data.LocalModelDownloadRecord
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import com.nousresearch.hermesagent.ui.chat.NativeToolCallingChatClient
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class HermesOnDevicePromptInstrumentedTest {
    private val app: Application
        get() = ApplicationProvider.getApplicationContext()

    @After
    fun tearDown() {
        HermesRuntimeManager.stop()
        OnDeviceBackendManager.stopAll()
    }

    @Test
    fun runGemma4PromptAndWriteResponseFile() {
        val arguments = InstrumentationRegistry.getArguments()
        val prompt = arguments.getString("hermes.prompt_b64")
            ?.let { encodedPrompt -> String(Base64.decode(encodedPrompt, Base64.DEFAULT), Charsets.UTF_8) }
            ?: arguments.getString("hermes.prompt")
            ?: "Reply with exactly HERMES_GEMMA_PROMPT_OK."
        val outputName = arguments.getString("hermes.output_name")
            ?.takeIf { it.isNotBlank() }
            ?: "hermes-gemma-prompt-output.json"
        val expectedForegroundPackage = arguments.getString("hermes.expected_foreground_package")
            ?.takeIf { it.isNotBlank() }
        require('/' !in outputName && '\\' !in outputName) {
            "hermes.output_name must be a simple file name"
        }

        val modelFile = File(app.filesDir, MODEL_RELATIVE_PATH)
        assumeTrue("Gemma 4 LiteRT-LM model is not provisioned at ${modelFile.absolutePath}", modelFile.isFile)
        seedPreferredGemma4Model(modelFile)

        val runtime = HermesRuntimeManager.ensureStarted(app)
        assertTrue(runtime.error.orEmpty(), runtime.started)
        val backendStatus = OnDeviceBackendManager.currentStatus()
        assertTrue(backendStatus.statusMessage, backendStatus.started)
        assertTrue(backendStatus.baseUrl, backendStatus.baseUrl.startsWith("http://127.0.0.1:"))

        val result = NativeToolCallingChatClient(app).send(
            baseUrl = backendStatus.baseUrl.removeSuffix("/v1"),
            modelName = backendStatus.modelName,
            sessionId = "gemma4-ad-hoc-phone-prompt",
            userText = prompt,
        )
        assertFalse("Expected a nonblank Gemma response", result.content.isBlank())
        val foregroundPackage = if (expectedForegroundPackage != null) {
            waitForForegroundPackage(expectedForegroundPackage)
        } else {
            currentForegroundPackage()
        }
        val outputFile = File(app.getExternalFilesDir(null), outputName)
        outputFile.writeText(
            JSONObject()
                .put("model", backendStatus.modelName)
                .put("backend", backendStatus.backendKind.name)
                .put("executed_tool_calls", result.executedToolCalls)
                .put("response", result.content)
                .put("foreground_package", foregroundPackage)
                .toString(2),
        )
        assertTrue("Expected prompt output at ${outputFile.absolutePath}", outputFile.isFile)
        if (expectedForegroundPackage != null) {
            assertTrue(
                "Expected foreground package $expectedForegroundPackage, got ${foregroundPackage.ifBlank { "<none>" }}",
                foregroundPackage == expectedForegroundPackage,
            )
        }
    }

    private fun waitForForegroundPackage(expectedPackage: String, timeoutMs: Long = 10_000): String {
        val deadline = SystemClock.uptimeMillis() + timeoutMs
        var latest = currentForegroundPackage()
        while (SystemClock.uptimeMillis() < deadline) {
            if (latest == expectedPackage) {
                return latest
            }
            SystemClock.sleep(250)
            latest = currentForegroundPackage().ifBlank { latest }
        }
        return latest
    }

    private fun currentForegroundPackage(): String {
        return InstrumentationRegistry.getInstrumentation()
            .uiAutomation
            .rootInActiveWindow
            ?.packageName
            ?.toString()
            .orEmpty()
    }

    private fun seedPreferredGemma4Model(modelFile: File) {
        val record = LocalModelDownloadRecord(
            id = "gemma-4-e2b-litertlm-ad-hoc-phone-prompt",
            title = MODEL_ID,
            sourceUrl = MODEL_SOURCE_URL,
            repoOrUrl = MODEL_REPO,
            filePath = MODEL_FILE_NAME,
            revision = MODEL_REVISION,
            runtimeFlavor = "LiteRT-LM",
            destinationFileName = MODEL_FILE_NAME,
            destinationPath = modelFile.absolutePath,
            downloadManagerId = -1L,
            totalBytes = MODEL_BYTES,
            downloadedBytes = MODEL_BYTES,
            status = "completed",
            statusMessage = "Provisioned for ad hoc phone prompt instrumentation",
            supportsResume = false,
        )
        LocalModelDownloadStore(app).apply {
            upsertDownload(record)
            setPreferredDownloadId(record.id)
        }
        AppSettingsStore(app).save(
            AppSettings(
                provider = "custom",
                baseUrl = "",
                model = MODEL_ID,
                onDeviceBackend = BackendKind.LITERT_LM.persistedValue,
            )
        )
    }

    private companion object {
        private const val MODEL_ID = "gemma-4-E2B-it"
        private const val MODEL_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
        private const val MODEL_FILE_NAME = "gemma-4-E2B-it.litertlm"
        private const val MODEL_RELATIVE_PATH = "hermes-home/downloads/models/$MODEL_FILE_NAME"
        private const val MODEL_SOURCE_URL =
            "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/7fa1d78473894f7e736a21d920c3aa80f950c0db/gemma-4-E2B-it.litertlm"
        private const val MODEL_REVISION = "7fa1d78473894f7e736a21d920c3aa80f950c0db"
        private const val MODEL_BYTES = 2_583_085_056L
    }
}
