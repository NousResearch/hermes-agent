_IMAGE_REJECTION_PHRASES = (
    "only 'text' content type is supported",
    "only text content type is supported",
    "image_url is not supported",
    "image content is not supported",
    "multimodal is not supported",
    "multimodal content is not supported",
    "multimodal input is not supported",
    "vision is not supported",
    "vision input is not supported",
    "does not support images",
    "does not support image input",
    "does not support multimodal",
    "does not support vision",
    "model does not support image",
    # ChatGPT-account Codex backend
    # (https://chatgpt.com/backend-api/codex) rejects
    # data:image/...base64 URLs in input_image fields
    # with HTTP 400 "Invalid 'input[N].content[K].image_url'.
    # Expected a valid URL, but got a value with an
    # invalid format." The OpenAI Responses API on the
    # public endpoint accepts data URLs, but the
    # ChatGPT-account variant does not. Without this
    # phrase the agent cascaded into compression /
    # context-too-large recovery instead of just
    # stripping the images. Match is narrow on
    # purpose — keyed on the field-path apostrophe so
    # we don't false-trip on other URL validation
    # errors. (issue #23570)
    "image_url'. expected",
    # DeepSeek's OpenAI-compatible API reports text-only
    # request-body variants as:
    # "unknown variant `image_url`, expected `text`".
    "unknown variant `image_url`, expected `text`",
    "unknown variant image_url, expected text",
    # OpenRouter routes a request to upstream endpoints and,
    # when none of the candidate endpoints for the model accept
    # image input, returns HTTP 404 "No endpoints found that
    # support image input". Without this phrase the agent never
    # strips the images, the retry loop re-sends the same
    # rejected request until exhaustion, and the gateway leaves
    # every subsequent message queued behind the stuck turn —
    # the P1 in issue #21160. The 404 passes the 4xx gate below.
    "no endpoints found that support image input",
)
