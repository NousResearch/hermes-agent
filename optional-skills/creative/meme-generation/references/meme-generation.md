# Meme Generation Source Notes

Imgflip is the canonical source for blank meme templates in this skill.

## Search and download workflow

1. Query the Imgflip template catalog.
2. Pick the best name or ID match.
3. Download the raw template image URL from Imgflip.
4. Use the downloaded file as the source image for captioning.

## Matching rules

The helper script ranks matches in this order:
- exact normalized name or ID
- prefix match
- substring match
- token overlap

## Common examples

- Absolute Cinema
  - Imgflip URL: https://i.imgflip.com/8d317n.png
- Woman Yelling at Cat
  - Imgflip URL: https://i.imgflip.com/345v97.jpg

## Notes

- The Imgflip API returns the template URL directly in the `url` field.
- Use the downloaded blank template, not a screenshot or a captioned copy.
- Keep the source image local before adding captions.
