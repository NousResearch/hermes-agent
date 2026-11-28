"""Offline MIME inspection: decode selected body/headers without visiting URLs.

Not a renderer, authenticity verifier, attachment executor or quote remover.
"""
import argparse
import json
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path


class TextAndLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text, self.links, self.hidden = [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        if tag == 'a':
            self.links.extend(v for k, v in attrs if k.lower() == 'href' and v)
        if tag in ('br', 'p', 'div', 'li', 'tr'):
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.hidden:
            self.hidden -= 1
        if tag in ('p', 'div', 'li', 'tr'):
            self.text.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


def inspect_mime(raw):
    if not isinstance(raw, bytes):
        raise ValueError('Supply raw RFC 5322 bytes, not a JSON wrapper')
    message = BytesParser(policy=policy.default).parsebytes(raw)
    defects = [type(d).__name__ for part in message.walk() for d in part.defects]
    result = {'headers': [{'name': name, 'value': str(value)} for name, value in message.items()],
              'body_status': 'unavailable', 'body_type': None, 'text': '', 'links': [],
              'warnings': defects, 'authentication_verified': False}
    if message.get_content_type() in ('multipart/encrypted', 'application/pkcs7-mime'):
        result['warnings'].append('Encrypted or enveloped message requires an appropriate decoder')
        return result
    body = message.get_body(preferencelist=('plain', 'html'))
    if body is None:
        return result
    try:
        content = body.get_content()
    except (LookupError, UnicodeError, ValueError) as exc:
        result['warnings'].append(type(exc).__name__)
        return result
    # Transfer decoding may add parser defects, e.g. invalid base64.
    defects = [type(d).__name__ for part in message.walk() for d in part.defects]
    result['warnings'] = list(dict.fromkeys(result['warnings'] + defects))
    if not isinstance(content, str) or not content.strip():
        return result
    if '\ufffd' in content:
        result['warnings'].append('Replacement characters indicate possibly lossy charset decoding')
    result['body_type'] = body.get_content_type()
    if result['body_type'] == 'text/html':
        parser = TextAndLinks()
        parser.feed(content)
        result['text'], result['links'] = ''.join(parser.text).strip(), parser.links
        result['warnings'].append('HTML text extraction is not visual rendering; hidden/quoted text may remain')
    else:
        result['text'] = content
        # Plain text is preferred for reading. Inspect its HTML alternative too
        # for actual link targets; do not substitute a larger attached message.
        html = message.get_body(preferencelist=('html',))
        if html is not None:
            try:
                parser = TextAndLinks()
                parser.feed(html.get_content())
                result['links'] = parser.links
            except (LookupError, UnicodeError, ValueError, TypeError):
                result['warnings'].append('HTML alternative could not be inspected')
    result['body_status'] = 'decoded_with_warnings' if result['warnings'] else 'decoded'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mime_file', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = inspect_mime(args.mime_file.read_bytes())
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'body_status': result['body_status'], 'warnings': len(result['warnings'])}))


if __name__ == '__main__':
    main()
