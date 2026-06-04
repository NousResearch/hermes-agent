# Rednote / Discord mirror publication note

Use `discord:#rednote` as the explicit target for Xiaohongshu/Rednote mirror sends in this workflow. If a prior send failed because the session was closed or the wrong Discord target was used, do not regenerate the image set; just resend the selected numbered items to `discord:#rednote`.

When the user says `发4到小红书` or `发布2-4到小红书`, preserve the requested selection exactly in the message bodies, but you may send the images in a separate batch and then give the user the completed delivery list.

If a single item in a batch fails to generate, keep the manifest and retry only that item. If publication happens across multiple sessions, rely on the manifest file paths rather than conversational recency.
