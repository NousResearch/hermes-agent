# Data Share Method

Open while preparing a file to leave the team. Where a team or regulator rule is stricter than this
page, that rule governs; say which one was used.

## 1. Field classes

A field's class comes from its content, not its name: a `notes` column holds emails, and a `ref`
column can hold an account number.

| Class | Examples | Default action |
| --- | --- | --- |
| direct identifier | name, email, phone number, postal address, customer or account number, national id, device id, IP address, user handle | drop; pseudonymize only when the destination must join rows |
| quasi-identifier | birth date, postcode, age, gender, job title, employer, exact timestamps, a rare value | generalize (year, region, band) or drop |
| secret | API key, token, password, session cookie, private key, connection string with credentials | drop; a secret found in the data is also a leak to rotate |
| free text | notes, comments, ticket bodies, transcript excerpts, log messages, URLs with query strings | drop, or keep only after the scan counts it clean |
| not personal | product id, count, price, status code | keep when the destination needs it |

## 2. Minimization order

1. Drop every field the destination did not ask for.
2. Generalize the quasi-identifiers that stay: a date to its month or year, a postcode to its region,
   an age to a band.
3. Pseudonymize a direct identifier only when rows must stay joinable.
4. Mask what must stay partly readable (the last four digits), knowing a masked value is still a
   quasi-identifier.

## 3. Pseudonymization

- Use a keyed hash (HMAC-SHA256 with a key made for this share — `data_share.py pseudonymize`) or a
  random token table. A plain hash of an email or a phone number is reversible by hashing candidate
  values.
- The key or the table is the mapping. Keep it at a local path outside the shipped file and its
  folder; never commit, attach or upload it.
- A new destination gets a new key, so two shared files cannot be joined against each other.

## 4. Residual scan, counted per class

Run it on the exact file that leaves, after the last transform, and print counts only. A count of `0`
is a result; a scan that did not run is `not_observed`.

| Class | What to count | `scan` field |
| --- | --- | --- |
| direct identifier | email shapes, phone shapes, card numbers that pass Luhn, IP addresses, and each original value of a dropped or pseudonymized identifier column matched in the output | `totals`, `original_values` |
| secret | vendor key prefixes, bearer tokens, JWTs, private key headers, password assignments, URLs with credentials | `totals.secret` |
| quasi-identifier | rows whose combination of kept quasi-identifiers is shared by fewer than k rows (k from the team; 5 is a common start) | `quasi_identifier` |
| free text | all of the above, inside each free-text field | `per_column` |

The original-value check is the one a pattern scan cannot replace: a pseudonymized email column
passes every email pattern while a notes field quotes the same address. It reads the source column
and the output file and prints one count per column; never type a value into a search pattern.

Verdict: `redacted` when every class counts zero, or each remaining hit is named and accepted by the
user; `not_observed` when no scan ran; otherwise not redacted, with the counts.

## 5. Share record

Destination, date, fields shipped, row count, the sha256 of the shipped file, and the scan result.
No value from the file, no mapping, no key.
