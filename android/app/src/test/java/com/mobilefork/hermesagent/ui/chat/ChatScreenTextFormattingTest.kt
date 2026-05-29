package com.mobilefork.hermesagent.ui.chat

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatScreenTextFormattingTest {
    @Test
    fun chatDisplayTextCleansMarkdownTablesAndEmphasis() {
        val rendered = sanitizeChatDisplayText(
            """
            **Hermes self-test**
            | Tool | Result | Detail |
            | --- | --- | --- |
            | terminal_tool | ready | bridge returned output |
            """.trimIndent(),
        )

        assertTrue(rendered.contains("Hermes self-test"))
        assertTrue(rendered.contains("Tool  Result  Detail"))
        assertTrue(rendered.contains("terminal_tool  ready  bridge returned output"))
        assertFalse(rendered.contains("**Hermes self-test**"))
        assertFalse(rendered.contains("| --- | --- | --- |"))
    }

    @Test
    fun chatDisplayTextCleansCollapsedInlineDiagnosticTable() {
        val rendered = sanitizeChatDisplayText(
            "**Full feature test results** | Tool | Status | Detail || --- | --- | --- || **Terminal** | X error | `env_var_enabled` not defined || **Android UI** | needs service | `HermesAccessibilityUiBridge` gated",
        )

        assertTrue(rendered.contains("Full feature test results  Tool  Status  Detail"))
        assertTrue(rendered.contains("Terminal  X error  env_var_enabled not defined"))
        assertTrue(rendered.contains("Android UI  needs service  HermesAccessibilityUiBridge gated"))
        assertFalse(rendered.contains("**"))
        assertFalse(rendered.contains("`"))
        assertFalse(rendered.contains("| --- |"))
    }

    @Test
    fun chatDisplayTextCleansTranslatedChineseFileDiagnostics() {
        val rendered = sanitizeChatDisplayText(
            "**看不到1.txt文件。 **经过搜索，目前环境中没有找到任何`.txt`文件。具体情况： | 位置 | 结果 || ----- | ----- || 当前目录 `/` | 无文件 || 用户家目录 | 无文件 || 全局搜索 `*.txt` | 0个结果 || Android共享文件夹 | 访问错误 **建议： **-请告诉我1.txt的**完整路径**，我直接尝试读取。",
        )

        assertTrue(rendered.contains("看不到1.txt文件。 经过搜索"))
        assertTrue(rendered.contains("当前目录 /  无文件"))
        assertTrue(rendered.contains("Android共享文件夹  访问错误 建议： -请告诉我1.txt的完整路径"))
        assertFalse(rendered.contains("**"))
        assertFalse(rendered.contains("`"))
        assertFalse(rendered.contains("||"))
        assertFalse(rendered.contains("-----"))
    }

    @Test
    fun chatDisplayTextCleansInlineLatexWrappers() {
        val rendered = sanitizeChatDisplayText("""Area is \(a^2 + b^2\), display \[E = mc^2\], and ${'$'}${'$'}x = y + z${'$'}${'$'}.""")

        assertTrue(rendered.contains("a^2 + b^2"))
        assertTrue(rendered.contains("E = mc^2"))
        assertTrue(rendered.contains("x = y + z"))
        assertFalse(rendered.contains("\\("))
        assertFalse(rendered.contains("\\["))
        assertFalse(rendered.contains("$$"))
    }

    @Test
    fun composerStatusCollapsesWhenKeyboardIsVisible() {
        assertTrue(shouldShowComposerStatus(tinyRuntimeViewport = false, imeVisible = false))
        assertFalse(shouldShowComposerStatus(tinyRuntimeViewport = true, imeVisible = false))
        assertFalse(shouldShowComposerStatus(tinyRuntimeViewport = false, imeVisible = true))
    }
}
