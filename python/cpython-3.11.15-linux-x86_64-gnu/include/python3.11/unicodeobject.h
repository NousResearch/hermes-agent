#ifndef Py_UNICODEOBJECT_H
#define Py_UNICODEOBJECT_H

#include <stdarg.h>               // va_list

/*

Unicode implementation based on original code by Fredrik Lundh,
modified by Marc-Andre Lemburg (mal@lemburg.com) according to the
Unicode Integration Proposal. (See
http://www.egenix.com/files/python/unicode-proposal.txt).

Copyright (c) Corporation for National Research Initiatives.


 Original header:
 --------------------------------------------------------------------

 * Yet another Unicode string type for Python.  This type supports the
 * 16-bit Basic Multilingual Plane (BMP) only.
 *
 * Written by Fredrik Lundh, January 1999.
 *
 * Copyright (c) 1999 by Secret Labs AB.
 * Copyright (c) 1999 by Fredrik Lundh.
 *
 * fredrik@pythonware.com
 * http://www.pythonware.com
 *
 * --------------------------------------------------------------------
 * This Unicode String Type is
 *
 * Copyright (c) 1999 by Secret Labs AB
 * Copyright (c) 1999 by Fredrik Lundh
 *
 * By obtaining, using, and/or copying this software and/or its
 * associated documentation, you agree that you have read, understood,
 * and will comply with the following terms and conditions:
 *
 * Permission to use, copy, modify, and distribute this software and its
 * associated documentation for any purpose and without fee is hereby
 * granted, provided that the above copyright notice appears in all
 * copies, and that both that copyright notice and this permission notice
 * appear in supporting documentation, and that the name of Secret Labs
 * AB or the author not be used in advertising or publicity pertaining to
 * distribution of the software without specific, written prior
 * permission.
 *
 * SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO
 * THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS.  IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
 * OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 * -------------------------------------------------------------------- */

#include <ctype.h>

/* === Internal API ======================================================= */

/* --- Internal Unicode Format -------------------------------------------- */

/* Python 3.x requires unicode */
#define Py_USING_UNICODE

#ifndef SIZEOF_WCHAR_T
#error Must define SIZEOF_WCHAR_T
#endif

#define Py_UNICODE_SIZE SIZEOF_WCHAR_T

/* If wchar_t can be used for UCS-4 storage, set Py_UNICODE_WIDE.
   Otherwise, Unicode strings are stored as UCS-2 (with limited support
   for UTF-16) */

#if Py_UNICODE_SIZE >= 4
#define Py_UNICODE_WIDE
#endif

/* Set these flags if the platform has "wchar.h" and the
   wchar_t type is a 16-bit unsigned type */
/* #define HAVE_WCHAR_H */
/* #define HAVE_USABLE_WCHAR_T */

/* If the compiler provides a wchar_t type we try to support it
   through the interface functions PyUnicode_FromWideChar(),
   PyUnicode_AsWideChar() and PyUnicode_AsWideCharString(). */

#ifdef HAVE_USABLE_WCHAR_T
# ifndef HAVE_WCHAR_H
#  define HAVE_WCHAR_H
# endif
#endif

#ifdef HAVE_WCHAR_H
#  include <wchar.h>
#endif

/* Py_UCS4 and Py_UCS2 are typedefs for the respective
   unicode representations. */
typedef uint32_t Py_UCS4;
typedef uint16_t Py_UCS2;
typedef uint8_t Py_UCS1;

#ifdef __cplusplus
extern "C" {
#endif


PyAPI_DATA(PyTypeObject) PyUnicode_Type;
PyAPI_DATA(PyTypeObject) PyUnicodeIter_Type;

#define PyUnicode_Check(op) \
    PyType_FastSubclass(Py_TYPE(op), Py_TPFLAGS_UNICODE_SUBCLASS)
#define PyUnicode_CheckExact(op) Py_IS_TYPE(op, &PyUnicode_Type)

/* --- Constants ---------------------------------------------------------- */

/* This Unicode character will be used as replacement character during
   decoding if the errors argument is set to "replace". Note: the
   Unicode character U+FFFD is the official REPLACEMENT CHARACTER in
   Unicode 3.0. */

#define Py_UNICODE_REPLACEMENT_CHARACTER ((Py_UCS4) 0xFFFD)

/* === Public API ========================================================= */

/* Similar to PyUnicode_FromUnicode(), but u points to UTF-8 encoded bytes */
PyAPI_FUNC(PyObject*) PyUnicode_FromStringAndSize(
    const char *u,             /* UTF-8 encoded string */
    Py_ssize_t size            /* size of buffer */
    );

/* Similar to PyUnicode_FromUnicode(), but u points to null-terminated
   UTF-8 encoded bytes.  The size is determined with strlen(). */
PyAPI_FUNC(PyObject*) PyUnicode_FromString(
    const char *u              /* UTF-8 encoded string */
    );

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
PyAPI_FUNC(PyObject*) PyUnicode_Substring(
    PyObject *str,
    Py_ssize_t start,
    Py_ssize_t end);
#endif

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
/* Copy the string into a UCS4 buffer including the null character if copy_null
   is set. Return NULL and raise an exception on error. Raise a SystemError if
   the buffer is smaller than the string. Return buffer on success.

   buflen is the length of the buffer in (Py_UCS4) characters. */
PyAPI_FUNC(Py_UCS4*) PyUnicode_AsUCS4(
    PyObject *unicode,
    Py_UCS4* buffer,
    Py_ssize_t buflen,
    int copy_null);

/* Copy the string into a UCS4 buffer. A new buffer is allocated using
 * PyMem_Malloc; if this fails, NULL is returned with a memory error
   exception set. */
PyAPI_FUNC(Py_UCS4*) PyUnicode_AsUCS4Copy(PyObject *unicode);
#endif

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
/* Get the length of the Unicode object. */

PyAPI_FUNC(Py_ssize_t) PyUnicode_GetLength(
    PyObject *unicode
);
#endif

/* Get the number of Py_UNICODE units in the
   string representation. */

Py_DEPRECATED(3.3) PyAPI_FUNC(Py_ssize_t) PyUnicode_GetSize(
    PyObject *unicode           /* Unicode object */
    );

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
/* Read a character from the string. */

PyAPI_FUNC(Py_UCS4) PyUnicode_ReadChar(
    PyObject *unicode,
    Py_ssize_t index
    );

/* Write a character to the string. The string must have been created through
   PyUnicode_New, must not be shared, and must not have been hashed yet.

   Return 0 on success, -1 on error. */

PyAPI_FUNC(int) PyUnicode_WriteChar(
    PyObject *unicode,
    Py_ssize_t index,
    Py_UCS4 character
    );
#endif

/* Resize a Unicode object. The length is the number of characters, except
   if the kind of the string is PyUnicode_WCHAR_KIND: in this case, the length
   is the number of Py_UNICODE characters.

   *unicode is modified to point to the new (resized) object and 0
   returned on success.

   Try to resize the string in place (which is usually faster than allocating
   a new string and copy characters), or create a new string.

   Error handling is implemented as follows: an exception is set, -1
   is returned and *unicode left untouched.

   WARNING: The function doesn't check string content, the result may not be a
            string in canonical representation. */

PyAPI_FUNC(int) PyUnicode_Resize(
    PyObject **unicode,         /* Pointer to the Unicode object */
    Py_ssize_t length           /* New length */
    );

/* Decode obj to a Unicode object.

   bytes, bytearray and other bytes-like objects are decoded according to the
   given encoding and error handler. The encoding and error handler can be
   NULL to have the interface use UTF-8 and "strict".

   All other objects (including Unicode objects) raise an exception.

   The API returns NULL in case of an error. The caller is responsible
   for decref'ing the returned objects.

*/

PyAPI_FUNC(PyObject*) PyUnicode_FromEncodedObject(
    PyObject *obj,              /* Object */
    const char *encoding,       /* encoding */
    const char *errors          /* error handling */
    );

/* Copy an instance of a Unicode subtype to a new true Unicode object if
   necessary. If obj is already a true Unicode object (not a subtype), return
   the reference with *incremented* refcount.

   The API returns NULL in case of an error. The caller is responsible
   for decref'ing the returned objects.

*/

PyAPI_FUNC(PyObject*) PyUnicode_FromObject(
    PyObject *obj      /* Object */
    );

PyAPI_FUNC(PyObject *) PyUnicode_FromFormatV(
    const char *format,   /* ASCII-encoded string  */
    va_list vargs
    );
PyAPI_FUNC(PyObject *) PyUnicode_FromFormat(
    const char *format,   /* ASCII-encoded string  */
    ...
    );

PyAPI_FUNC(void) PyUnicode_InternInPlace(PyObject **);
PyAPI_FUNC(PyObject *) PyUnicode_InternFromString(
    const char *u              /* UTF-8 encoded string */
    );

// PyUnicode_InternImmortal() is deprecated since Python 3.10
// and will be removed in Python 3.12. Use PyUnicode_InternInPlace() instead.
Py_DEPRECATED(3.10) PyAPI_FUNC(void) PyUnicode_InternImmortal(PyObject **);

/* --- wchar_t support for platforms which support it --------------------- */

#ifdef HAVE_WCHAR_H

/* Create a Unicode Object from the wchar_t buffer w of the given
   size.

   The buffer is copied into the new object. */

PyAPI_FUNC(PyObject*) PyUnicode_FromWideChar(
    const wchar_t *w,           /* wchar_t buffer */
    Py_ssize_t size             /* size of buffer */
    );

/* Copies the Unicode Object contents into the wchar_t buffer w.  At
   most size wchar_t characters are copied.

   Note that the resulting wchar_t string may or may not be
   0-terminated.  It is the responsibility of the caller to make sure
   that the wchar_t string is 0-terminated in case this is required by
   the application.

   Returns the number of wchar_t characters copied (excluding a
   possibly trailing 0-termination character) or -1 in case of an
   error. */

PyAPI_FUNC(Py_ssize_t) PyUnicode_AsWideChar(
    PyObject *unicode,          /* Unicode object */
    wchar_t *w,                 /* wchar_t buffer */
    Py_ssize_t size             /* size of buffer */
    );

/* Convert the Unicode object to a wide character string. The output string
   always ends with a nul character. If size is not NULL, write the number of
   wide characters (excluding the null character) into *size.

   Returns a buffer allocated by PyMem_Malloc() (use PyMem_Free() to free it)
   on success. On error, returns NULL, *size is undefined and raises a
   MemoryError. */

PyAPI_FUNC(wchar_t*) PyUnicode_AsWideCharString(
    PyObject *unicode,          /* Unicode object */
    Py_ssize_t *size            /* number of characters of the result */
    );

#endif

/* --- Unicode ordinals --------------------------------------------------- */

/* Create a Unicode Object from the given Unicode code point ordinal.

   The ordinal must be in range(0x110000). A ValueError is
   raised in case it is not.

*/

PyAPI_FUNC(PyObject*) PyUnicode_FromOrdinal(int ordinal);

/* === Builtin Codecs =====================================================

   Many of these APIs take two arguments encoding and errors. These
   parameters encoding and errors have the same semantics as the ones
   of the builtin str() API.

   Setting encoding to NULL causes the default encoding (UTF-8) to be used.

   Error handling is set by errors which may also be set to NULL
   meaning to use the default handling defined for the codec. Default
   error handling for all builtin codecs is "strict" (ValueErrors are
   raised).

   The codecs all use a similar interface. Only deviation from the
   generic ones are documented.

*/

/* --- Manage the default encoding ---------------------------------------- */

/* Returns "utf-8".  */
PyAPI_FUNC(const char*) PyUnicode_GetDefaultEncoding(void);

/* --- Generic Codecs ----------------------------------------------------- */

/* Create a Unicode object by decoding the encoded string s of the
   given size. */

PyAPI_FUNC(PyObject*) PyUnicode_Decode(
    const char *s,              /* encoded string */
    Py_ssize_t size,            /* size of buffer */
    const char *encoding,       /* encoding */
    const char *errors          /* error handling */
    );

/* Decode a Unicode object unicode and return the result as Python
   object.

   This API is DEPRECATED. The only supported standard encoding is rot13.
   Use PyCodec_Decode() to decode with rot13 and non-standard codecs
   that decode from str. */

Py_DEPRECATED(3.6) PyAPI_FUNC(PyObject*) PyUnicode_AsDecodedObject(
    PyObject *unicode,          /* Unicode object */
    const char *encoding,       /* encoding */
    const char *errors          /* error handling */
    );

/* Decode a Unicode object unicode and return the result as Unicode
   object.

   This API is DEPRECATED. The only supported standard encoding is rot13.
   Use PyCodec_Decode() to decode with rot13 and non-standard codecs
   that decode from str to str. */

Py_DEPRECATED(3.6) PyAPI_FUNC(PyObject*) PyUnicode_AsDecodedUnicode(
    PyObject *unicode,          /* Unicode object */
    const char *encoding,       /* encoding */
    const char *errors          /* error handling */
    );

/* Encodes a Unicode object and returns the result as Python
   object.

   This API is DEPRECATED.  It is superseded by PyUnicode_AsEncodedString()
   since all standard encodings (except rot13) encode str to bytes.
   Use PyCodec_Encode() for encoding with rot13 and non-standard codecs
   that encode form str to non-bytes. */

Py_DEPRECATED(3.6) PyAPI_FUNC(PyObject*) PyUnicode_AsEncodedObject(
    PyObject *unicode,          /* Unicode object */
    const char *encoding,       /* encoding */
    const char *errors          /* error handling */
    );

/* Encodes a Unicode object and returns the result as Python string
   object. */

PyAPI_FUNC(PyObject*) PyUnicode_AsEncodedString(
    PyObject *unicode,          /* Unicode object */
    const char *encoding,       /* encoding */
    const char *errors          /* error handling */
    );

/* Encodes a Unicode object and returns the result as Unicode
   object.

   This API is DEPRECATED.  The only supported standard encodings is rot13.
   Use PyCodec_Encode() to encode with rot13 and non-standard codecs
   that encode from str to str. */

Py_DEPRECATED(3.6) PyAPI_FUNC(PyObject*) PyUnicode_AsEncodedUnicode(
    PyObject *unicode,          /* Unicode object */
    const char *encoding,       /* encoding */
    const char *errors          /* error handling */
    );

/* Build an encoding map. */

PyAPI_FUNC(PyObject*) PyUnicode_BuildEncodingMap(
    PyObject* string            /* 256 character map */
   );

/* --- UTF-7 Codecs ------------------------------------------------------- */

PyAPI_FUNC(PyObject*) PyUnicode_DecodeUTF7(
    const char *string,         /* UTF-7 encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors          /* error handling */
    );

PyAPI_FUNC(PyObject*) PyUnicode_DecodeUTF7Stateful(
    const char *string,         /* UTF-7 encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors,         /* error handling */
    Py_ssize_t *consumed        /* bytes consumed */
    );

/* --- UTF-8 Codecs ------------------------------------------------------- */

PyAPI_FUNC(PyObject*) PyUnicode_DecodeUTF8(
    const char *string,         /* UTF-8 encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors          /* error handling */
    );

PyAPI_FUNC(PyObject*) PyUnicode_DecodeUTF8Stateful(
    const char *string,         /* UTF-8 encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors,         /* error handling */
    Py_ssize_t *consumed        /* bytes consumed */
    );

PyAPI_FUNC(PyObject*) PyUnicode_AsUTF8String(
    PyObject *unicode           /* Unicode object */
    );

/* Returns a pointer to the default encoding (UTF-8) of the
   Unicode object unicode and the size of the encoded representation
   in bytes stored in *size.

   In case of an error, no *size is set.

   This function caches the UTF-8 encoded string in the unicodeobject
   and subsequent calls will return the same string.  The memory is released
   when the unicodeobject is deallocated.
*/

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x030A0000
PyAPI_FUNC(const char *) PyUnicode_AsUTF8AndSize(
    PyObject *unicode,
    Py_ssize_t *size);
#endif

/* --- UTF-32 Codecs ------------------------------------------------------ */

/* Decodes length bytes from a UTF-32 encoded buffer string and returns
   the corresponding Unicode object.

   errors (if non-NULL) defines the error handling. It defaults
   to "strict".

   If byteorder is non-NULL, the decoder starts decoding using the
   given byte order:

    *byteorder == -1: little endian
    *byteorder == 0:  native order
    *byteorder == 1:  big endian

   In native mode, the first four bytes of the stream are checked for a
   BOM mark. If found, the BOM mark is analysed, the byte order
   adjusted and the BOM skipped.  In the other modes, no BOM mark
   interpretation is done. After completion, *byteorder is set to the
   current byte order at the end of input data.

   If byteorder is NULL, the codec starts in native order mode.

*/

PyAPI_FUNC(PyObject*) PyUnicode_DecodeUTF32(
    const char *string,         /* UTF-32 encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors,         /* error handling */
    int *byteorder              /* pointer to byteorder to use
                                   0=native;-1=LE,1=BE; updated on
                                   exit */
    );

PyAPI_FUNC(PyObject*) PyUnicode_DecodeUTF32Stateful(
    const char *string,         /* UTF-32 encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors,         /* error handling */
    int *byteorder,             /* pointer to byteorder to use
                                   0=native;-1=LE,1=BE; updated on
                                   exit */
    Py_ssize_t *consumed        /* bytes consumed */
    );

/* Returns a Python string using the UTF-32 encoding in native byte
   order. The string always starts with a BOM mark.  */

PyAPI_FUNC(PyObject*) PyUnicode_AsUTF32String(
    PyObject *unicode           /* Unicode object */
    );

/* Returns a Python string object holding the UTF-32 encoded value of
   the Unicode data.

   If byteorder is not 0, output is written according to the following
   byte order:

   byteorder == -1: little endian
   byteorder == 0:  native byte order (writes a BOM mark)
   byteorder == 1:  big endian

   If byteorder is 0, the output string will always start with the
   Unicode BOM mark (U+FEFF). In the other two modes, no BOM mark is
   prepended.

*/

/* --- UTF-16 Codecs ------------------------------------------------------ */

/* Decodes length bytes from a UTF-16 encoded buffer string and returns
   the corresponding Unicode object.

   errors (if non-NULL) defines the error handling. It defaults
   to "strict".

   If byteorder is non-NULL, the decoder starts decoding using the
   given byte order:

    *byteorder == -1: little endian
    *byteorder == 0:  native order
    *byteorder == 1:  big endian

   In native mode, the first two bytes of the stream are checked for a
   BOM mark. If found, the BOM mark is analysed, the byte order
   adjusted and the BOM skipped.  In the other modes, no BOM mark
   interpretation is done. After completion, *byteorder is set to the
   current byte order at the end of input data.

   If byteorder is NULL, the codec starts in native order mode.

*/

PyAPI_FUNC(PyObject*) PyUnicode_DecodeUTF16(
    const char *string,         /* UTF-16 encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors,         /* error handling */
    int *byteorder              /* pointer to byteorder to use
                                   0=native;-1=LE,1=BE; updated on
                                   exit */
    );

PyAPI_FUNC(PyObject*) PyUnicode_DecodeUTF16Stateful(
    const char *string,         /* UTF-16 encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors,         /* error handling */
    int *byteorder,             /* pointer to byteorder to use
                                   0=native;-1=LE,1=BE; updated on
                                   exit */
    Py_ssize_t *consumed        /* bytes consumed */
    );

/* Returns a Python string using the UTF-16 encoding in native byte
   order. The string always starts with a BOM mark.  */

PyAPI_FUNC(PyObject*) PyUnicode_AsUTF16String(
    PyObject *unicode           /* Unicode object */
    );

/* --- Unicode-Escape Codecs ---------------------------------------------- */

PyAPI_FUNC(PyObject*) PyUnicode_DecodeUnicodeEscape(
    const char *string,         /* Unicode-Escape encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors          /* error handling */
    );

PyAPI_FUNC(PyObject*) PyUnicode_AsUnicodeEscapeString(
    PyObject *unicode           /* Unicode object */
    );

/* --- Raw-Unicode-Escape Codecs ------------------------------------------ */

PyAPI_FUNC(PyObject*) PyUnicode_DecodeRawUnicodeEscape(
    const char *string,         /* Raw-Unicode-Escape encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors          /* error handling */
    );

PyAPI_FUNC(PyObject*) PyUnicode_AsRawUnicodeEscapeString(
    PyObject *unicode           /* Unicode object */
    );

/* --- Latin-1 Codecs -----------------------------------------------------

   Note: Latin-1 corresponds to the first 256 Unicode ordinals. */

PyAPI_FUNC(PyObject*) PyUnicode_DecodeLatin1(
    const char *string,         /* Latin-1 encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors          /* error handling */
    );

PyAPI_FUNC(PyObject*) PyUnicode_AsLatin1String(
    PyObject *unicode           /* Unicode object */
    );

/* --- ASCII Codecs -------------------------------------------------------

   Only 7-bit ASCII data is excepted. All other codes generate errors.

*/

PyAPI_FUNC(PyObject*) PyUnicode_DecodeASCII(
    const char *string,         /* ASCII encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors          /* error handling */
    );

PyAPI_FUNC(PyObject*) PyUnicode_AsASCIIString(
    PyObject *unicode           /* Unicode object */
    );

/* --- Character Map Codecs -----------------------------------------------

   This codec uses mappings to encode and decode characters.

   Decoding mappings must map byte ordinals (integers in the range from 0 to
   255) to Unicode strings, integers (which are then interpreted as Unicode
   ordinals) or None.  Unmapped data bytes (ones which cause a LookupError)
   as well as mapped to None, 0xFFFE or '\ufffe' are treated as "undefined
   mapping" and cause an error.

   Encoding mappings must map Unicode ordinal integers to bytes objects,
   integers in the range from 0 to 255 or None.  Unmapped character
   ordinals (ones which cause a LookupError) as well as mapped to
   None are treated as "undefined mapping" and cause an error.

*/

PyAPI_FUNC(PyObject*) PyUnicode_DecodeCharmap(
    const char *string,         /* Encoded string */
    Py_ssize_t length,          /* size of string */
    PyObject *mapping,          /* decoding mapping */
    const char *errors          /* error handling */
    );

PyAPI_FUNC(PyObject*) PyUnicode_AsCharmapString(
    PyObject *unicode,          /* Unicode object */
    PyObject *mapping           /* encoding mapping */
    );

/* --- MBCS codecs for Windows -------------------------------------------- */

#ifdef MS_WINDOWS
PyAPI_FUNC(PyObject*) PyUnicode_DecodeMBCS(
    const char *string,         /* MBCS encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors          /* error handling */
    );

PyAPI_FUNC(PyObject*) PyUnicode_DecodeMBCSStateful(
    const char *string,         /* MBCS encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors,         /* error handling */
    Py_ssize_t *consumed        /* bytes consumed */
    );

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
PyAPI_FUNC(PyObject*) PyUnicode_DecodeCodePageStateful(
    int code_page,              /* code page number */
    const char *string,         /* encoded string */
    Py_ssize_t length,          /* size of string */
    const char *errors,         /* error handling */
    Py_ssize_t *consumed        /* bytes consumed */
    );
#endif

PyAPI_FUNC(PyObject*) PyUnicode_AsMBCSString(
    PyObject *unicode           /* Unicode object */
    );

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
PyAPI_FUNC(PyObject*) PyUnicode_EncodeCodePage(
    int code_page,              /* code page number */
    PyObject *unicode,          /* Unicode object */
    const char *errors          /* error handling */
    );
#endif

#endif /* MS_WINDOWS */

/* --- Locale encoding --------------------------------------------------- */

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
/* Decode a string from the current locale encoding. The decoder is strict if
   *surrogateescape* is equal to zero, otherwise it uses the 'surrogateescape'
   error handler (PEP 383) to escape undecodable bytes. If a byte sequence can
   be decoded as a surrogate character and *surrogateescape* is not equal to
   zero, the byte sequence is escaped using the 'surrogateescape' error handler
   instead of being decoded. *str* must end with a null character but cannot
   contain embedded null characters. */

PyAPI_FUNC(PyObject*) PyUnicode_DecodeLocaleAndSize(
    const char *str,
    Py_ssize_t len,
    const char *errors);

/* Similar to PyUnicode_DecodeLocaleAndSize(), but compute the string
   length using strlen(). */

PyAPI_FUNC(PyObject*) PyUnicode_DecodeLocale(
    const char *str,
    const char *errors);

/* Encode a Unicode object to the current locale encoding. The encoder is
   strict is *surrogateescape* is equal to zero, otherwise the
   "surrogateescape" error handler is used. Return a bytes object. The string
   cannot contain embedded null characters. */

PyAPI_FUNC(PyObject*) PyUnicode_EncodeLocale(
    PyObject *unicode,
    const char *errors
    );
#endif

/* --- File system encoding ---------------------------------------------- */

/* ParseTuple converter: encode str objects to bytes using
   PyUnicode_EncodeFSDefault(); bytes objects are output as-is. */

PyAPI_FUNC(int) PyUnicode_FSConverter(PyObject*, void*);

/* ParseTuple converter: decode bytes objects to unicode using
   PyUnicode_DecodeFSDefaultAndSize(); str objects are output as-is. */

PyAPI_FUNC(int) PyUnicode_FSDecoder(PyObject*, void*);

/* Decode a null-terminated string using Py_FileSystemDefaultEncoding
   and the "surrogateescape" error handler.

   If Py_FileSystemDefaultEncoding is not set, fall back to the locale
   encoding.

   Use PyUnicode_DecodeFSDefaultAndSize() if the string length is known.
*/

PyAPI_FUNC(PyObject*) PyUnicode_DecodeFSDefault(
    const char *s               /* encoded string */
    );

/* Decode a string using Py_FileSystemDefaultEncoding
   and the "surrogateescape" error handler.

   If Py_FileSystemDefaultEncoding is not set, fall back to the locale
   encoding.
*/

PyAPI_FUNC(PyObject*) PyUnicode_DecodeFSDefaultAndSize(
    const char *s,               /* encoded string */
    Py_ssize_t size              /* size */
    );

/* Encode a Unicode object to Py_FileSystemDefaultEncoding with the
   "surrogateescape" error handler, and return bytes.

   If Py_FileSystemDefaultEncoding is not set, fall back to the locale
   encoding.
*/

PyAPI_FUNC(PyObject*) PyUnicode_EncodeFSDefault(
    PyObject *unicode
    );

/* --- Methods & Slots ----------------------------------------------------

   These are capable of handling Unicode objects and strings on input
   (we refer to them as strings in the descriptions) and return
   Unicode objects or integers as appropriate. */

/* Concat two strings giving a new Unicode string. */

PyAPI_FUNC(PyObject*) PyUnicode_Concat(
    PyObject *left,             /* Left string */
    PyObject *right             /* Right string */
    );

/* Concat two strings and put the result in *pleft
   (sets *pleft to NULL on error) */

PyAPI_FUNC(void) PyUnicode_Append(
    PyObject **pleft,           /* Pointer to left string */
    PyObject *right             /* Right string */
    );

/* Concat two strings, put the result in *pleft and drop the right object
   (sets *pleft to NULL on error) */

PyAPI_FUNC(void) PyUnicode_AppendAndDel(
    PyObject **pleft,           /* Pointer to left string */
    PyObject *right             /* Right string */
    );

/* Split a string giving a list of Unicode strings.

   If sep is NULL, splitting will be done at all whitespace
   substrings. Otherwise, splits occur at the given separator.

   At most maxsplit splits will be done. If negative, no limit is set.

   Separators are not included in the resulting list.

*/

PyAPI_FUNC(PyObject*) PyUnicode_Split(
    PyObject *s,                /* String to split */
    PyObject *sep,              /* String separator */
    Py_ssize_t maxsplit         /* Maxsplit count */
    );

/* Dito, but split at line breaks.

   CRLF is considered to be one line break. Line breaks are not
   included in the resulting list. */

PyAPI_FUNC(PyObject*) PyUnicode_Splitlines(
    PyObject *s,                /* String to split */
    int keepends                /* If true, line end markers are included */
    );

/* Partition a string using a given separator. */

PyAPI_FUNC(PyObject*) PyUnicode_Partition(
    PyObject *s,                /* String to partition */
    PyObject *sep               /* String separator */
    );

/* Partition a string using a given separator, searching from the end of the
   string. */

PyAPI_FUNC(PyObject*) PyUnicode_RPartition(
    PyObject *s,                /* String to partition */
    PyObject *sep               /* String separator */
    );

/* Split a string giving a list of Unicode strings.

   If sep is NULL, splitting will be done at all whitespace
   substrings. Otherwise, splits occur at the given separator.

   At most maxsplit splits will be done. But unlike PyUnicode_Split
   PyUnicode_RSplit splits from the end of the string. If negative,
   no limit is set.

   Separators are not included in the resulting list.

*/

PyAPI_FUNC(PyObject*) PyUnicode_RSplit(
    PyObject *s,                /* String to split */
    PyObject *sep,              /* String separator */
    Py_ssize_t maxsplit         /* Maxsplit count */
    );

/* Translate a string by applying a character mapping table to it and
   return the resulting Unicode object.

   The mapping table must map Unicode ordinal integers to Unicode strings,
   Unicode ordinal integers or None (causing deletion of the character).

   Mapping tables may be dictionaries or sequences. Unmapped character
   ordinals (ones which cause a LookupError) are left untouched and
   are copied as-is.

*/

PyAPI_FUNC(PyObject *) PyUnicode_Translate(
    PyObject *str,              /* String */
    PyObject *table,            /* Translate table */
    const char *errors          /* error handling */
    );

/* Join a sequence of strings using the given separator and return
   the resulting Unicode string. */

PyAPI_FUNC(PyObject*) PyUnicode_Join(
    PyObject *separator,        /* Separator string */
    PyObject *seq               /* Sequence object */
    );

/* Return 1 if substr matches str[start:end] at the given tail end, 0
   otherwise. */

PyAPI_FUNC(Py_ssize_t) PyUnicode_Tailmatch(
    PyObject *str,              /* String */
    PyObject *substr,           /* Prefix or Suffix string */
    Py_ssize_t start,           /* Start index */
    Py_ssize_t end,             /* Stop index */
    int direction               /* Tail end: -1 prefix, +1 suffix */
    );

/* Return the first position of substr in str[start:end] using the
   given search direction or -1 if not found. -2 is returned in case
   an error occurred and an exception is set. */

PyAPI_FUNC(Py_ssize_t) PyUnicode_Find(
    PyObject *str,              /* String */
    PyObject *substr,           /* Substring to find */
    Py_ssize_t start,           /* Start index */
    Py_ssize_t end,             /* Stop index */
    int direction               /* Find direction: +1 forward, -1 backward */
    );

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
/* Like PyUnicode_Find, but search for single character only. */
PyAPI_FUNC(Py_ssize_t) PyUnicode_FindChar(
    PyObject *str,
    Py_UCS4 ch,
    Py_ssize_t start,
    Py_ssize_t end,
    int direction
    );
#endif

/* Count the number of occurrences of substr in str[start:end]. */

PyAPI_FUNC(Py_ssize_t) PyUnicode_Count(
    PyObject *str,              /* String */
    PyObject *substr,           /* Substring to count */
    Py_ssize_t start,           /* Start index */
    Py_ssize_t end              /* Stop index */
    );

/* Replace at most maxcount occurrences of substr in str with replstr
   and return the resulting Unicode object. */

PyAPI_FUNC(PyObject *) PyUnicode_Replace(
    PyObject *str,              /* String */
    PyObject *substr,           /* Substring to find */
    PyObject *replstr,          /* Substring to replace */
    Py_ssize_t maxcount         /* Max. number of replacements to apply;
                                   -1 = all */
    );

/* Compare two strings and return -1, 0, 1 for less than, equal,
   greater than resp.
   Raise an exception and return -1 on error. */

PyAPI_FUNC(int) PyUnicode_Compare(
    PyObject *left,             /* Left string */
    PyObject *right             /* Right string */
    );

/* Compare a Unicode object with C string and return -1, 0, 1 for less than,
   equal, and greater than, respectively.  It is best to pass only
   ASCII-encoded strings, but the function interprets the input string as
   ISO-8859-1 if it contains non-ASCII characters.
   This function does not raise exceptions. */

PyAPI_FUNC(int) PyUnicode_CompareWithASCIIString(
    PyObject *left,
    const char *right           /* ASCII-encoded string */
    );

/* Rich compare two strings and return one of the following:

   - NULL in case an exception was raised
   - Py_True or Py_False for successful comparisons
   - Py_NotImplemented in case the type combination is unknown

   Possible values for op:

     Py_GT, Py_GE, Py_EQ, Py_NE, Py_LT, Py_LE

*/

PyAPI_FUNC(PyObject *) PyUnicode_RichCompare(
    PyObject *left,             /* Left string */
    PyObject *right,            /* Right string */
    int op                      /* Operation: Py_EQ, Py_NE, Py_GT, etc. */
    );

/* Apply an argument tuple or dictionary to a format string and return
   the resulting Unicode string. */

PyAPI_FUNC(PyObject *) PyUnicode_Format(
    PyObject *format,           /* Format string */
    PyObject *args              /* Argument tuple or dictionary */
    );

/* Checks whether element is contained in container and return 1/0
   accordingly.

   element has to coerce to a one element Unicode string. -1 is
   returned in case of an error. */

PyAPI_FUNC(int) PyUnicode_Contains(
    PyObject *container,        /* Container string */
    PyObject *element           /* Element string */
    );

/* Checks whether argument is a valid identifier. */

PyAPI_FUNC(int) PyUnicode_IsIdentifier(PyObject *s);

/* === Characters Type APIs =============================================== */

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_UNICODEOBJECT_H
#  include "cpython/unicodeobject.h"
#  undef Py_CPYTHON_UNICODEOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_UNICODEOBJECT_H */
