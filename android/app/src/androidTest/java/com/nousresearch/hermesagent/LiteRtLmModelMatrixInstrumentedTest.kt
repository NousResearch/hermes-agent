package com.nousresearch.hermesagent

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.nousresearch.hermesagent.backend.LiteRtLmOpenAiProxy
import com.nousresearch.hermesagent.backend.OnDeviceBackendManager
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
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class LiteRtLmModelMatrixInstrumentedTest {
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
    fun provisionedLiteRtLmModelLoadsAndAnswersLocally() {
        val args = InstrumentationRegistry.getArguments()
        val modelId = args.getString("model_id", DEFAULT_MODEL_ID)
        val modelFileName = args.getString("model_file_name", DEFAULT_MODEL_FILE_NAME)
        val expectedBytes = args.getString("model_bytes", DEFAULT_MODEL_BYTES.toString()).toLong()
        val modelFile = File(context.filesDir, "hermes-home/downloads/models/$modelFileName")

        assumeTrue("LiteRT-LM model is not provisioned at ${modelFile.absolutePath}", modelFile.isFile)
        if (expectedBytes > 0L) {
            assertEquals("$modelId LiteRT-LM model size", expectedBytes, modelFile.length())
        }

        val status = LiteRtLmOpenAiProxy.ensureRunning(
            context = context,
            modelPath = modelFile.absolutePath,
            requestedModelName = modelId,
            port = OnDeviceBackendManager.LITERT_LM_PORT,
        )
        assertTrue(status.statusMessage, status.started)

        val health = executeJson(
            Request.Builder()
                .url(status.baseUrl.removeSuffix("/v1") + "/health")
                .get()
                .build()
        )
        assertEquals(health.toString(), "ok", health.optString("status"))
        assertEquals(health.toString(), "litert-lm", health.optString("backend"))

        val completion = executeJson(
            Request.Builder()
                .url("${status.baseUrl}/chat/completions")
                .post(completionRequestBody(modelId))
                .build()
        )
        val content = completion
            .getJSONArray("choices")
            .getJSONObject(0)
            .getJSONObject("message")
            .optString("content")
        assertFalse(completion.toString(), content.isBlank())
    }

    private fun completionRequestBody(modelId: String) = JSONObject()
        .put("model", modelId)
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
        private const val DEFAULT_MODEL_ID = "gemma-4-E2B-it"
        private const val DEFAULT_MODEL_FILE_NAME = "gemma-4-E2B-it.litertlm"
        private const val DEFAULT_MODEL_BYTES = 2_583_085_056L
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }
}
