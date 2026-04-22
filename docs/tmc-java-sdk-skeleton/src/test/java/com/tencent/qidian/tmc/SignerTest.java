package com.tencent.qidian.tmc;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.tencent.qidian.tmc.auth.HmacSha256ApiSigner;
import com.tencent.qidian.tmc.auth.HmacSha256WebhookSigner;
import org.junit.jupiter.api.Test;

/**
 * Basic signer tests.
 */
public class SignerTest {
    @Test
    void shouldGenerateApiSign() {
        HmacSha256ApiSigner signer = new HmacSha256ApiSigner();
        String sign = signer.sign("corp_1", "1711111111111", "secret");
        assertNotNull(sign);
        assertTrue(!sign.isEmpty());
    }

    @Test
    void shouldVerifyWebhookSign() {
        HmacSha256WebhookSigner signer = new HmacSha256WebhookSigner();
        String sign = signer.signCallback("https://example.com/callback", "1711111111111", "secret");
        assertTrue(signer.verify("https://example.com/callback", "1711111111111", sign, "secret"));
        assertFalse(signer.verify("https://example.com/callback", "1711111111112", sign, "secret"));
    }
}
