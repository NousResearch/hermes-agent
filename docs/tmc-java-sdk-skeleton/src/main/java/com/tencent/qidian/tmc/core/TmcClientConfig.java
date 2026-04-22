package com.tencent.qidian.tmc.core;

/**
 * TMC SDK client configuration.
 */
public final class TmcClientConfig {
    private final String baseUrl;
    private final String corporationId;
    private final String secretId;
    private final String secretKey;
    private final int connectTimeoutMillis;
    private final int readTimeoutMillis;
    private final boolean enableLogging;

    private TmcClientConfig(Builder builder) {
        this.baseUrl = builder.baseUrl;
        this.corporationId = builder.corporationId;
        this.secretId = builder.secretId;
        this.secretKey = builder.secretKey;
        this.connectTimeoutMillis = builder.connectTimeoutMillis;
        this.readTimeoutMillis = builder.readTimeoutMillis;
        this.enableLogging = builder.enableLogging;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    public String getCorporationId() {
        return corporationId;
    }

    public String getSecretId() {
        return secretId;
    }

    public String getSecretKey() {
        return secretKey;
    }

    public int getConnectTimeoutMillis() {
        return connectTimeoutMillis;
    }

    public int getReadTimeoutMillis() {
        return readTimeoutMillis;
    }

    public boolean isEnableLogging() {
        return enableLogging;
    }

    public static final class Builder {
        private String baseUrl = "https://tmc.qidian.qq.com";
        private String corporationId;
        private String secretId;
        private String secretKey;
        private int connectTimeoutMillis = 3000;
        private int readTimeoutMillis = 10000;
        private boolean enableLogging;

        public Builder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        public Builder corporationId(String corporationId) {
            this.corporationId = corporationId;
            return this;
        }

        public Builder secretId(String secretId) {
            this.secretId = secretId;
            return this;
        }

        public Builder secretKey(String secretKey) {
            this.secretKey = secretKey;
            return this;
        }

        public Builder connectTimeoutMillis(int connectTimeoutMillis) {
            this.connectTimeoutMillis = connectTimeoutMillis;
            return this;
        }

        public Builder readTimeoutMillis(int readTimeoutMillis) {
            this.readTimeoutMillis = readTimeoutMillis;
            return this;
        }

        public Builder enableLogging(boolean enableLogging) {
            this.enableLogging = enableLogging;
            return this;
        }

        public TmcClientConfig build() {
            return new TmcClientConfig(this);
        }
    }
}
