package com.tencent.qidian.tmc.core;

import java.util.Map;

/**
 * Low-level HTTP abstraction for TMC API calls.
 */
public interface TmcHttpClient {
    <T> T post(String path, Object request, Class<T> responseType);

    <T> T get(String path, Map<String, Object> query, Class<T> responseType);
}
