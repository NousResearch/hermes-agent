package com.tencent.qidian.tmc.core;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.tencent.qidian.tmc.auth.ApiSigner;
import com.tencent.qidian.tmc.exception.TmcAuthException;
import com.tencent.qidian.tmc.exception.TmcBadRequestException;
import com.tencent.qidian.tmc.exception.TmcRemoteException;
import com.tencent.qidian.tmc.model.common.ApiResponse;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Map;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * OkHttp-based default HTTP client implementation.
 */
public class OkHttpTmcHttpClient implements TmcHttpClient {
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");

    private final TmcClientConfig config;
    private final OkHttpClient okHttpClient;
    private final ObjectMapper objectMapper;
    private final ApiSigner apiSigner;

    public OkHttpTmcHttpClient(TmcClientConfig config,
                               OkHttpClient okHttpClient,
                               ObjectMapper objectMapper,
                               ApiSigner apiSigner) {
        this.config = config;
        this.okHttpClient = okHttpClient;
        this.objectMapper = objectMapper.copy()
                .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        this.apiSigner = apiSigner;
    }

    @Override
    public <T> T post(String path, Object request, Class<T> responseType) {
        validateConfig();
        try {
            String body = objectMapper.writeValueAsString(request);
            String timestamp = timestamp();
            Request httpRequest = new Request.Builder()
                    .url(config.getBaseUrl() + normalizePath(path))
                    .post(RequestBody.create(body, JSON))
                    .addHeader("corporationId", config.getCorporationId())
                    .addHeader("timestamp", timestamp)
                    .addHeader("sign", sign(timestamp))
                    .addHeader("secretId", emptyToBlank(config.getSecretId()))
                    .build();
            return execute(httpRequest, responseType);
        } catch (IOException e) {
            throw new TmcBadRequestException("Failed to serialize request.", null, null);
        }
    }

    @Override
    public <T> T get(String path, Map<String, Object> query, Class<T> responseType) {
        validateConfig();
        String timestamp = timestamp();
        Request httpRequest = new Request.Builder()
                .url(buildUrl(path, query))
                .get()
                .addHeader("corporationId", config.getCorporationId())
                .addHeader("timestamp", timestamp)
                .addHeader("sign", sign(timestamp))
                .addHeader("secretId", emptyToBlank(config.getSecretId()))
                .build();
        return execute(httpRequest, responseType);
    }

    private <T> T execute(Request request, Class<T> responseType) {
        try (Response response = okHttpClient.newCall(request).execute()) {
            String json = response.body() == null ? "" : response.body().string();
            if (!response.isSuccessful()) {
                throw new TmcRemoteException("HTTP invocation failed with status " + response.code(), response.code(), null);
            }
            JavaType envelopeType = objectMapper.getTypeFactory()
                    .constructParametricType(ApiResponse.class, responseType);
            ApiResponse<T> envelope = objectMapper.readValue(json, envelopeType);
            if (envelope.getCode() != null && envelope.getCode().intValue() != 0) {
                throw new TmcRemoteException(envelope.getMessage(), envelope.getCode(), envelope.getRequestId());
            }
            return envelope.getData();
        } catch (IOException e) {
            throw new TmcRemoteException("HTTP invocation failed.", null, null);
        }
    }

    private void validateConfig() {
        if (isBlank(config.getCorporationId()) || isBlank(config.getSecretKey())) {
            throw new TmcAuthException("corporationId and secretKey must be provided.");
        }
    }

    private String buildUrl(String path, Map<String, Object> query) {
        StringBuilder builder = new StringBuilder(config.getBaseUrl()).append(normalizePath(path));
        if (query != null && !query.isEmpty()) {
            boolean first = true;
            for (Map.Entry<String, Object> entry : query.entrySet()) {
                if (entry.getValue() == null) {
                    continue;
                }
                builder.append(first ? '?' : '&');
                first = false;
                builder.append(encode(entry.getKey())).append('=').append(encode(String.valueOf(entry.getValue())));
            }
        }
        return builder.toString();
    }

    private String normalizePath(String path) {
        return path.startsWith("/") ? path : "/" + path;
    }

    private String encode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8);
    }

    private String timestamp() {
        return String.valueOf(Instant.now().toEpochMilli());
    }

    private String sign(String timestamp) {
        return apiSigner.sign(config.getCorporationId(), timestamp, config.getSecretKey());
    }

    private boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    private String emptyToBlank(String value) {
        return value == null ? "" : value;
    }
}
