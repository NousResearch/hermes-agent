package com.tencent.qidian.tmc.webhook;

import java.util.Map;

/**
 * Minimal HTTP-agnostic webhook request wrapper.
 */
public class WebhookRequest {
    private final String requestUrl;
    private final Map<String, String> queryParams;
    private final String rawBody;

    public WebhookRequest(String requestUrl, Map<String, String> queryParams, String rawBody) {
        this.requestUrl = requestUrl;
        this.queryParams = queryParams;
        this.rawBody = rawBody;
    }

    public String getRequestUrl() {
        return requestUrl;
    }

    public Map<String, String> getQueryParams() {
        return queryParams;
    }

    public String getRawBody() {
        return rawBody;
    }
}
