import sys
import unittest
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mime_inspect import inspect_mime


class MimeTests(unittest.TestCase):
    def test_plain_body_wins_over_larger_attachment(self):
        msg = EmailMessage()
        msg['Subject'] = 'Invoice'
        msg.set_content('Actual receipt body', charset='utf-8', cte='base64')
        msg.add_attachment(('Unrelated quoted promotion ' * 200).encode(), maintype='text', subtype='plain', filename='attached.txt')
        result = inspect_mime(msg.as_bytes())
        self.assertIn('Actual receipt body', result['text'])
        self.assertNotIn('Unrelated', result['text'])
        self.assertEqual(result['body_status'], 'decoded')

    def test_html_links_are_data_and_not_visited(self):
        msg = EmailMessage()
        msg.set_content('<p>Hello &amp; goodbye</p><script>secret()</script><a href="https://example.invalid/path">click</a>', subtype='html')
        result = inspect_mime(msg.as_bytes())
        self.assertIn('Hello & goodbye', result['text'])
        self.assertNotIn('secret()', result['text'])
        self.assertEqual(result['links'], ['https://example.invalid/path'])
        self.assertFalse(result['authentication_verified'])

    def test_plain_alternative_keeps_html_link_targets(self):
        msg = EmailMessage()
        msg.set_content('Text version')
        msg.add_alternative('<a href="https://example.invalid/real-target">Different brand</a>', subtype='html')
        result = inspect_mime(msg.as_bytes())
        self.assertIn('Text version', result['text'])
        self.assertEqual(result['links'], ['https://example.invalid/real-target'])

    def test_folded_repeated_and_encoded_headers(self):
        raw = (b'Subject: =?utf-8?q?Invoice_=E2=82=AC?=\r\n'
               b'List-Unsubscribe: <mailto:list@example.invalid>,\r\n <https://example.invalid/u>\r\n'
               b'Authentication-Results: first; dkim=pass\r\n'
               b'Authentication-Results: second; dkim=fail\r\n\r\nBody')
        result = inspect_mime(raw)
        headers = result['headers']
        self.assertEqual(headers[0]['value'], 'Invoice €')
        self.assertIn('https://example.invalid/u', headers[1]['value'])
        self.assertEqual(len([h for h in headers if h['name'] == 'Authentication-Results']), 2)
        self.assertFalse(result['authentication_verified'])

    def test_encrypted_body_stays_unavailable(self):
        result = inspect_mime(b'Content-Type: application/pkcs7-mime\r\n\r\nopaque')
        self.assertEqual(result['body_status'], 'unavailable')

    def test_unknown_charset_stays_unavailable(self):
        result = inspect_mime(b'Content-Type: text/plain; charset=unknown-xyz\r\n\r\nhello')
        self.assertEqual(result['body_status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
