# Elements, motion library, and media uploads

Use this reference for library operations and media preparation. The live `tools/list` and `who_am_i` responses define whether a tool is available, its arguments, input names, and resource limits.

## Elements

- To browse reusable subjects, call `element_list` and show the returned `id` and `name`. The list may not include the resource type; call `element_get(id)` before reuse, update, or generation. If a name is ambiguous, show candidates and ask the user to choose. Do not guess an ID.
- To create an Element, require an explicit request to save a reusable subject. Collect its name, description, tags, and resources, then validate the complete resource object against the MCP contract before calling `element_create`. Do not create an Element automatically for an ordinary reference-image generation.
- To update an Element, call `element_get` first and merge the requested changes into the complete `name`, `description`, `resource`, and `tags`. Preserve fields the user did not change, including the existing image `resource.cover`. Treat `secondary[]` as a full replacement: preserve unmodified secondary images and keep the provider's 1–3 image constraint when declared.
- Do not silently convert an image Element into a video Element or vice versa. If replacing a cover requires delete-and-recreate, explain that the ID will change and obtain explicit authorization before deleting.
- Call `element_delete` only after the user explicitly identifies the Element to delete. A read-only library operation does not require a credit check or generation submission.

## Using Elements in generation

1. With a known ID, call `element_get`. With only a name, call `element_list`, resolve the choice, then call `element_get`.
2. Confirm that the current account can access the Element and inspect `resource` to determine whether it is an image or video resource.
3. Use an Element only when both the tool and selected model explicitly support `elements`. Text-to-image and text-to-video do not use Elements. A video Element is not an image substitute.
4. Bind the exact Element ID in both places: include `<<<ELEMENT_ID>>>` in the prompt where the subject is referenced and pass `elements` as the model-declared argument, with a JSON-array string such as `[{"id":"ELEMENT_ID","bindName":"subject"}]`. Do not use a placeholder, omit the structured binding, or invent a different field shape.
5. An Element does not replace required image inputs. Supply `image_1`, `first_image`, or another exact live input when the selected model requires it. For motion control, the subject `image` remains required.
6. After resolving the Element, hand the request to the image or video generation skill and follow its confirmation, credit, single-submission, and status rules.

## Motion library and motion control

- To browse saved motions, call `motion_library_list` and show the real name, `id`, preview when present, duration, and audio availability. Durations returned in milliseconds may be displayed in seconds by dividing by 1000. An empty list is a valid result; do not invent motions.
- Selecting a motion by name requires an unambiguous match. If names collide or the intended motion is unclear, ask the user to choose. Browsing the library never calls `motion_control`.
- The current tool surface provides no motion create, update, or delete operation. Do not use `element_*` tools to manage motions.
- For motion control, pass the subject image as the live `image` input. Use exactly one motion source: the library ID as the `motionId` argument or a source video as the live `video` input. Never pass a preview URL as `motionId`, and do not pass both sources.
- Call `who_am_i` for the selected `motion_control` model and use only its declared `motionDirection`, `resolution`, and `keepOriginalSound` names and values. When `motionDirection=image_direction`, the motion source must satisfy the live 3–10 second constraint; do not pass the library duration as an undeclared `duration` argument.
- Forward the resolved request to the video generation skill. Credit checks, confirmation, one submission, and querying by `generationId` still apply.

## Local media uploads

- Prefer media already attached by Hermes and accepted by the selected model. If a public URL is accepted by the live tool, use it directly; do not download it unnecessarily. A historical Kling result must be refreshed by `generationId` immediately before reuse, never reused from an old URL in conversation history.
- If no usable attachment reference exists and the live MCP exposes `file_upload`, request an upload ticket with the real filename, MIME type, byte size, and the same `taskTraceId`. Never send a local path as a remote URL.
- Complete the provider's second upload step with the returned ticket and upload URL using multipart form data containing `ticket` and the binary `file`. A ticket response is not upload completion; use the media URL only after the upload service confirms success.
- Tickets are single-use and expire at the provider-declared `expireAt`. Never log or display tickets, signed upload URLs, credentials, cookies, or authorization headers.
- If the host cannot read the file or the live MCP lacks upload support, explain the limitation and ask the user to attach usable media. Do not submit a generation or write an Element that depends on unavailable media. Do not loop on failed ticket requests or uploads.
