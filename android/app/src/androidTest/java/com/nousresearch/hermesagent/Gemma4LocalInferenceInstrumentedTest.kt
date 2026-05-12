package com.nousresearch.hermesagent

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.nousresearch.hermesagent.backend.BackendKind
import com.nousresearch.hermesagent.backend.LiteRtLmOpenAiProxy
import com.nousresearch.hermesagent.backend.OnDeviceBackendManager
import com.nousresearch.hermesagent.data.AppSettings
import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.data.LocalModelDownloadRecord
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import java.io.File
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class Gemma4LocalInferenceInstrumentedTest {
    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.MINUTES)
        .build()

    @After
    fun tearDown() {
        OnDeviceBackendManager.stopAll()
    }

    @Test
    fun aGemma4LiteRtLmLoadsAndAnswersLocally() {
        val modelFile = File(context.filesDir, MODEL_RELATIVE_PATH)
        assumeTrue("Gemma 4 LiteRT-LM model is not provisioned at ${modelFile.absolutePath}", modelFile.isFile)
        assertEquals("Gemma 4 LiteRT-LM model size", MODEL_BYTES, modelFile.length())

        seedPreferredGemma4Model(modelFile)

        val status = OnDeviceBackendManager.ensureConfigured(
            context = context,
            backendValue = BackendKind.LITERT_LM.persistedValue,
        )
        assertTrue(status.statusMessage, status.started)
        assertEquals(BackendKind.LITERT_LM, status.backendKind)
        assertEquals(modelFile.absolutePath, status.sourceModelPath)
        assertTrue(status.baseUrl, status.baseUrl.startsWith("http://127.0.0.1:"))

        val healthUrl = status.baseUrl.removeSuffix("/v1") + "/health"
        val health = executeJson(Request.Builder().url(healthUrl).get().build())
        assertEquals(health.toString(), "ok", health.optString("status"))
        assertEquals(health.toString(), "litert-lm", health.optString("backend"))
        assertTrue(health.toString(), health.has("image_input_supported"))
        assertTrue(health.toString(), health.has("audio_input_supported"))
        assertTrue(health.toString(), health.has("modality_policy"))
        if (health.optBoolean("multimodal_fallback", false)) {
            assertFalse(health.toString(), health.optBoolean("image_input_supported", true))
            assertFalse(health.toString(), health.optBoolean("audio_input_supported", true))
            val modalityPolicy = health.optString("modality_policy")
            assertTrue(
                health.toString(),
                modalityPolicy.contains("text-only fallback") ||
                    modalityPolicy.contains("text-only memory guard"),
            )
        }

        val completion = executeJson(
            Request.Builder()
                .url("${status.baseUrl}/chat/completions")
                .post(completionRequestBody())
                .build()
        )
        val choices = completion.getJSONArray("choices")
        assertTrue(completion.toString(), choices.length() > 0)
        val content = choices.getJSONObject(0).getJSONObject("message").optString("content")
        assertFalse(completion.toString(), content.isBlank())
    }

    @Test
    fun bDirectLiteRtLmProxyCanServeProvisionedGemma4Model() {
        val modelFile = File(context.filesDir, MODEL_RELATIVE_PATH)
        assumeTrue("Gemma 4 LiteRT-LM model is not provisioned at ${modelFile.absolutePath}", modelFile.isFile)
        assertEquals("Gemma 4 LiteRT-LM model size", MODEL_BYTES, modelFile.length())

        val status = LiteRtLmOpenAiProxy.ensureRunning(
            context = context,
            modelPath = modelFile.absolutePath,
            requestedModelName = MODEL_ID,
            port = OnDeviceBackendManager.LITERT_LM_PORT,
        )
        assertTrue(status.statusMessage, status.started)

        val completion = executeJson(
            Request.Builder()
                .url("${status.baseUrl}/chat/completions")
                .post(completionRequestBody())
                .build()
        )
        val content = completion
            .getJSONArray("choices")
            .getJSONObject(0)
            .getJSONObject("message")
            .optString("content")
        assertFalse(completion.toString(), content.isBlank())
    }

    private fun seedPreferredGemma4Model(modelFile: File) {
        val record = LocalModelDownloadRecord(
            id = "gemma-4-e2b-litertlm-local-smoke",
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
            statusMessage = "Provisioned for local instrumentation",
            supportsResume = false,
        )
        LocalModelDownloadStore(context).apply {
            upsertDownload(record)
            setPreferredDownloadId(record.id)
        }
        AppSettingsStore(context).save(
            AppSettings(
                provider = "custom",
                baseUrl = "",
                model = MODEL_ID,
                onDeviceBackend = BackendKind.LITERT_LM.persistedValue,
            )
        )
    }

    private fun completionRequestBody() = JSONObject()
        .put("model", MODEL_ID)
        .put(
            "messages",
            JSONArray().put(
                JSONObject()
                    .put("role", "user")
                    .put("content", "Reply with exactly one short word: ok")
            )
        )
        .put("stream", false)
        .toString()
        .toRequestBody(JSON_MEDIA_TYPE)

    private fun executeJson(request: Request): JSONObject {
        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            assertTrue(body, response.isSuccessful)
            return JSONObject(body)
        }
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
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }
}
