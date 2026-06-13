// Rebases relative A1-style references inside formula strings by a row/column offset.
// The model writes in-table formulas as if the table's top-left cell were A1
// (table-local); the broker shifts them to the real start_cell before writing, so
// =B2*C2 written at Sheet1!H23 becomes =I24*J24. Absolute halves ($B$2, B$2, $B2)
// and sheet-qualified refs (Sheet1!B5, 'Bank Rec'!A1) are never shifted.
//
// This is the SINGLE rebasing layer (broker-side, in normalizeActions). The pane
// writes action.values verbatim — never add a second layer, or formulas shift twice.
// Imported by broker/server.mjs and broker/server.test.mjs.

const MAX_COLUMNS = 16384; // XFD
const MAX_ROWS = 1048576;

// Optional sheet qualifier, then a cell ref or range with independent $ anchors.
// The trailing lookahead rejects function names (LOG10() and partial identifiers;
// the leading lookbehind rejects matches inside longer names or after a sheet "!".
const REF_PATTERN =
  /(?<![\w$.!:])((?:'(?:[^']|'')*'|[A-Za-z_][\w.]*)!)?(\$?)([A-Za-z]{1,3})(\$?)(\d{1,7})(?::(\$?)([A-Za-z]{1,3})(\$?)(\d{1,7}))?(?![\w(])/g;

export function columnLettersToNumber(letters) {
  const text = String(letters || "").toUpperCase();
  if (!text) return 0;
  let result = 0;
  for (const char of text) {
    const code = char.charCodeAt(0);
    if (code < 65 || code > 90) return 0;
    result = result * 26 + (code - 64);
  }
  return result;
}

export function numberToColumnLetters(n) {
  let value = Number(n);
  if (!Number.isInteger(value) || value < 1) return "";
  let result = "";
  while (value > 0) {
    result = String.fromCharCode(65 + ((value - 1) % 26)) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

export function anchorFromAddress(ref) {
  const text = String(ref || "");
  const local = text.slice(text.lastIndexOf("!") + 1);
  const match = local.match(/^\s*\$?([A-Za-z]{1,3})\$?(\d{1,7})/);
  if (!match) return { rowOffset: 0, colOffset: 0 };
  const column = columnLettersToNumber(match[1]);
  const row = Number(match[2]);
  if (column < 1 || column > MAX_COLUMNS || row < 1 || row > MAX_ROWS) {
    return { rowOffset: 0, colOffset: 0 };
  }
  return { rowOffset: row - 1, colOffset: column - 1 };
}

function shiftEndpoint(columnAnchor, letters, rowAnchor, digits, rowOffset, colOffset) {
  const column = columnLettersToNumber(letters);
  const row = Number(digits);
  // A "column" past XFD is a defined name, not a ref; row 0 is not a ref either.
  if (column < 1 || column > MAX_COLUMNS || row < 1 || row > MAX_ROWS) return null;
  const shiftedColumn = columnAnchor ? column : column + colOffset;
  const shiftedRow = rowAnchor ? row : row + rowOffset;
  if (shiftedColumn < 1 || shiftedColumn > MAX_COLUMNS || shiftedRow < 1 || shiftedRow > MAX_ROWS) return null;
  return `${columnAnchor}${numberToColumnLetters(shiftedColumn)}${rowAnchor}${shiftedRow}`;
}

function shiftRefs(code, rowOffset, colOffset) {
  return code.replace(
    REF_PATTERN,
    (match, sheet, columnAnchor1, letters1, rowAnchor1, digits1, columnAnchor2, letters2, rowAnchor2, digits2) => {
      if (sheet) return match;
      const first = shiftEndpoint(columnAnchor1, letters1, rowAnchor1, digits1, rowOffset, colOffset);
      if (first === null) return match;
      if (letters2 === undefined) return first;
      const second = shiftEndpoint(columnAnchor2, letters2, rowAnchor2, digits2, rowOffset, colOffset);
      if (second === null) return match;
      return `${first}:${second}`;
    },
  );
}

export function translateFormula(text, rowOffset, colOffset) {
  if (typeof text !== "string" || !text.startsWith("=")) return text;
  if (!rowOffset && !colOffset) return text;

  // Copy double-quoted string literals through untouched ("" is the escaped quote).
  let out = "";
  let index = 0;
  while (index < text.length) {
    if (text[index] === '"') {
      let end = index + 1;
      while (end < text.length) {
        if (text[end] === '"') {
          if (text[end + 1] === '"') {
            end += 2;
            continue;
          }
          end += 1;
          break;
        }
        end += 1;
      }
      out += text.slice(index, end);
      index = end;
    } else {
      let end = text.indexOf('"', index);
      if (end === -1) end = text.length;
      out += shiftRefs(text.slice(index, end), rowOffset, colOffset);
      index = end;
    }
  }
  return out;
}

export function translateMatrixFormulas(matrix, rowOffset, colOffset) {
  if (!Array.isArray(matrix)) return matrix;
  return matrix.map((row) =>
    Array.isArray(row) ? row.map((cell) => translateFormula(cell, rowOffset, colOffset)) : row,
  );
}
