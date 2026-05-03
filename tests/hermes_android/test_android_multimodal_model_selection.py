from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_android_embedded_python_runtime_is_upgraded_to_312_in_ci_and_gradle():
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")
    push_workflow = (REPO_ROOT / ".github/workflows/android.yml").read_text(encoding="utf-8")
    release_workflow = (REPO_ROOT / ".github/workflows/android-release.yml").read_text(encoding="utf-8")

    assert 'version = "3.12"' in gradle
    assert '"python3.12"' in gradle
    assert "python-version: '3.12'" in push_workflow
    assert "python-version: '3.12'" in release_workflow
    assert "java-version: '21'" in push_workflow
    assert "java-version: '21'" in release_workflow
    assert 'command -v python3.12' in push_workflow
    assert 'command -v python3.12' in release_workflow


def test_settings_model_selection_uses_real_dropdown_card_with_gemma3_and_gemma4_options():
    settings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    presets = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/ProviderPresets.kt").read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")

    assert 'ModelSelectionCard(' in settings
    assert 'ExposedDropdownMenuBox(expanded = expanded' in settings
    assert 'HermesModelDropdown' in settings
    assert 'localizedOnDeviceSummary(summary, strings)' in settings
    assert 'trimmed.startsWith("No preferred local model")' in settings
    assert 'ProviderPresets.modelSelections(providerId)' in settings
    assert 'Gemma 4 E2B (LiteRT-LM)' in presets
    assert 'Gemma 4 E4B (LiteRT-LM)' in presets
    assert 'Gemma 3 1B IT INT4 (LiteRT-LM)' in presets
    assert 'Gemma 3 4B IT Vision (.task)' in presets
    assert 'Gemma 3n E2B IT Vision (LiteRT-LM)' in presets
    assert 'modelSelectionTitle' in strings
    assert 'first-class local Gemma 4, Gemma 3, and Gemma 3n models' in strings
    assert 'Aún no hay un modelo local compatible seleccionado' in strings


def test_chat_multimodal_request_path_attaches_images_as_openai_content_parts():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    chat_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatViewModel.kt").read_text(encoding="utf-8")
    api_models = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/api/HermesApiModels.kt").read_text(encoding="utf-8")
    native_client = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")

    assert 'ActivityResultContracts.OpenDocument()' in chat_screen
    assert 'imageLauncher.launch(arrayOf("image/*"))' in chat_screen
    assert 'attachImage(uri.toString())' in chat_screen
    assert 'data class ChatAttachment' in (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatState.kt").read_text(encoding="utf-8")
    assert 'ChatContentPart(' in chat_view_model
    assert 'type = "image_url"' in chat_view_model
    assert 'data:$mimeType;base64,' in chat_view_model
    assert 'val contentParts: List<ChatContentPart>' in api_models
    assert 'put("type", "image_url")' in api_models
    assert 'userContentParts: List<ChatContentPart>' in native_client


def test_litert_lm_proxy_accepts_image_content_for_vision_models_and_rejects_text_only_models():
    proxy = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/LiteRtLmOpenAiProxy.kt").read_text(encoding="utf-8")
    backend = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/OnDeviceBackendManager.kt").read_text(encoding="utf-8")

    assert 'Content.ImageBytes(Base64.decode' in proxy
    assert 'Content.ImageFile' in proxy
    assert 'requestContainsImage(requestMessages) && !supportsImageInput' in proxy
    assert 'maxNumImages = if (supportImage) 1 else null' in proxy
    assert 'visionBackend = visionBackend' in proxy
    assert 'openClAvailable -> Backend.GPU()' in proxy
    assert 'preferred.supportsImageInput()' in backend
    assert '"gemma-3n" in lower' in backend
    assert '"gemma3-4b" in lower' in backend

    matrix_test = (REPO_ROOT / "android/app/src/androidTest/java/com/nousresearch/hermesagent/LiteRtLmModelMatrixInstrumentedTest.kt").read_text(encoding="utf-8")
    assert 'provisionedVisionLiteRtLmModelDescribesImageLocally' in matrix_test
    assert 'provisionedTextOnlyLiteRtLmModelRejectsImageRequestsClearly' in matrix_test
    assert 'image input requires a LiteRT-LM model started with image support' in matrix_test
