const assert = require('assert');
const { contentText } = require('../src/acpClient');

assert.strictEqual(contentText('hello'), 'hello');
assert.strictEqual(contentText({ type: 'text', text: 'hello' }), 'hello');
assert.strictEqual(contentText([{ type: 'text', text: 'a' }, { type: 'text', text: 'b' }]), 'ab');
assert.strictEqual(contentText({ content: { type: 'text', text: 'nested' } }), 'nested');
assert.strictEqual(contentText({ resource: { text: 'resource text' } }), 'resource text');
assert.strictEqual(contentText(null), '');
console.log('acpClient tests ok');
