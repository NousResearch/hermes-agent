package com.nousresearch.hermesagent.models

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class HermesModelDownloadManagerTest {
    @Test
    fun liteRtLmArtifactSelection_prefersMtpLiteRtLmBundleOverWebTask() {
        assertTrue(isCompatibleRepoFile("gemma-4-E2B-it.litertlm", "LiteRT-LM"))
        assertFalse(isCompatibleRepoFile("gemma-4-E2B-it-web.task", "LiteRT-LM"))
        assertEquals(0, compatibleFileRank("gemma-4-E2B-it.litertlm", "LiteRT-LM"))
        assertEquals(Int.MAX_VALUE, compatibleFileRank("gemma-4-E2B-it-web.task", "LiteRT-LM"))
    }

    @Test
    fun liteRtLmArtifactSelection_keepsGenericGemma4MtpBundleAheadOfSoCSpecificVariants() {
        assertTrue(isCompatibleRepoFile("gemma-4-E2B-it_qualcomm_sm8750.litertlm", "LiteRT-LM"))
        assertTrue(
            compatibleFileRank("gemma-4-E2B-it.litertlm", "LiteRT-LM") <
                compatibleFileRank("gemma-4-E2B-it_qualcomm_sm8750.litertlm", "LiteRT-LM")
        )
    }

    @Test
    fun liteRtLmAliasesPinEdgeGalleryMtpRevisionsForGemma4() {
        assertEquals(
            "litert-community/gemma-4-E2B-it-litert-lm",
            liteRtAlias("google/gemma-4-e2b-it"),
        )
        assertEquals(
            "litert-community/gemma-4-E4B-it-litert-lm",
            liteRtAlias("google/gemma-4-e4b-it"),
        )
        assertEquals(
            "7fa1d78473894f7e736a21d920c3aa80f950c0db",
            liteRtAliasRevision("litert-community/gemma-4-E2B-it-litert-lm"),
        )
        assertEquals(
            "9695417f248178c63a9f318c6e0c56cb917cb837",
            liteRtAliasRevision("litert-community/gemma-4-E4B-it-litert-lm"),
        )
    }

    private fun isCompatibleRepoFile(path: String, runtimeFlavor: String): Boolean {
        return callPrivate("isCompatibleRepoFile", path, runtimeFlavor) as Boolean
    }

    private fun compatibleFileRank(path: String, runtimeFlavor: String): Int {
        return callPrivate("compatibleFileRank", path, runtimeFlavor) as Int
    }

    private fun liteRtAlias(repoId: String): String? {
        return callPrivate("liteRtAlias", repoId) as String?
    }

    private fun liteRtAliasRevision(repoId: String): String? {
        val field = HermesModelDownloadManager::class.java.getDeclaredField("LITERT_ALIAS_REVISIONS")
        field.isAccessible = true
        @Suppress("UNCHECKED_CAST")
        val revisions = field.get(HermesModelDownloadManager) as Map<String, String>
        return revisions[repoId.lowercase()]
    }

    private fun callPrivate(name: String, vararg args: Any): Any? {
        val argTypes = args.map { it::class.java }.toTypedArray()
        val method = HermesModelDownloadManager::class.java.getDeclaredMethod(name, *argTypes)
        method.isAccessible = true
        return method.invoke(HermesModelDownloadManager, *args)
    }
}
