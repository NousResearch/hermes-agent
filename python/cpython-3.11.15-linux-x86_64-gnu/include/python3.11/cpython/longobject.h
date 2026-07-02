#ifndef Py_CPYTHON_LONGOBJECT_H
#  error "this header file must not be included directly"
#endif

PyAPI_FUNC(int) _PyLong_AsInt(PyObject *);

PyAPI_FUNC(int) _PyLong_UnsignedShort_Converter(PyObject *, void *);
PyAPI_FUNC(int) _PyLong_UnsignedInt_Converter(PyObject *, void *);
PyAPI_FUNC(int) _PyLong_UnsignedLong_Converter(PyObject *, void *);
PyAPI_FUNC(int) _PyLong_UnsignedLongLong_Converter(PyObject *, void *);
PyAPI_FUNC(int) _PyLong_Size_t_Converter(PyObject *, void *);

/* _PyLong_Frexp returns a double x and an exponent e such that the
   true value is approximately equal to x * 2**e.  e is >= 0.  x is
   0.0 if and only if the input is 0 (in which case, e and x are both
   zeroes); otherwise, 0.5 <= abs(x) < 1.0.  On overflow, which is
   possible if the number of bits doesn't fit into a Py_ssize_t, sets
   OverflowError and returns -1.0 for x, 0 for e. */
PyAPI_FUNC(double) _PyLong_Frexp(PyLongObject *a, Py_ssize_t *e);

PyAPI_FUNC(PyObject *) PyLong_FromUnicodeObject(PyObject *u, int base);
PyAPI_FUNC(PyObject *) _PyLong_FromBytes(const char *, Py_ssize_t, int);

/* _PyLong_Sign.  Return 0 if v is 0, -1 if v < 0, +1 if v > 0.
   v must not be NULL, and must be a normalized long.
   There are no error cases.
*/
PyAPI_FUNC(int) _PyLong_Sign(PyObject *v);

/* _PyLong_NumBits.  Return the number of bits needed to represent the
   absolute value of a long.  For example, this returns 1 for 1 and -1, 2
   for 2 and -2, and 2 for 3 and -3.  It returns 0 for 0.
   v must not be NULL, and must be a normalized long.
   (size_t)-1 is returned and OverflowError set if the true result doesn't
   fit in a size_t.
*/
PyAPI_FUNC(size_t) _PyLong_NumBits(PyObject *v);

/* _PyLong_DivmodNear.  Given integers a and b, compute the nearest
   integer q to the exact quotient a / b, rounding to the nearest even integer
   in the case of a tie.  Return (q, r), where r = a - q*b.  The remainder r
   will satisfy abs(r) <= abs(b)/2, with equality possible only if q is
   even.
*/
PyAPI_FUNC(PyObject *) _PyLong_DivmodNear(PyObject *, PyObject *);

/* _PyLong_FromByteArray:  View the n unsigned bytes as a binary integer in
   base 256, and return a Python int with the same numeric value.
   If n is 0, the integer is 0.  Else:
   If little_endian is 1/true, bytes[n-1] is the MSB and bytes[0] the LSB;
   else (little_endian is 0/false) bytes[0] is the MSB and bytes[n-1] the
   LSB.
   If is_signed is 0/false, view the bytes as a non-negative integer.
   If is_signed is 1/true, view the bytes as a 2's-complement integer,
   non-negative if bit 0x80 of the MSB is clear, negative if set.
   Error returns:
   + Return NULL with the appropriate exception set if there's not
     enough memory to create the Python int.
*/
PyAPI_FUNC(PyObject *) _PyLong_FromByteArray(
    const unsigned char* bytes, size_t n,
    int little_endian, int is_signed);

/* _PyLong_AsByteArray: Convert the least-significant 8*n bits of long
   v to a base-256 integer, stored in array bytes.  Normally return 0,
   return -1 on error.
   If little_endian is 1/true, store the MSB at bytes[n-1] and the LSB at
   bytes[0]; else (little_endian is 0/false) store the MSB at bytes[0] and
   the LSB at bytes[n-1].
   If is_signed is 0/false, it's an error if v < 0; else (v >= 0) n bytes
   are filled and there's nothing special about bit 0x80 of the MSB.
   If is_signed is 1/true, bytes is filled with the 2's-complement
   representation of v's value.  Bit 0x80 of the MSB is the sign bit.
   Error returns (-1):
   + is_signed is 0 and v < 0.  TypeError is set in this case, and bytes
     isn't altered.
   + n isn't big enough to hold the full mathematical value of v.  For
     example, if is_signed is 0 and there are more digits in the v than
     fit in n; or if is_signed is 1, v < 0, and n is just 1 bit shy of
     being large enough to hold a sign bit.  OverflowError is set in this
     case, but bytes holds the least-significant n bytes of the true value.
*/
PyAPI_FUNC(int) _PyLong_AsByteArray(PyLongObject* v,
    unsigned char* bytes, size_t n,
    int little_endian, int is_signed);

/* _PyLong_Format: Convert the long to a string object with given base,
   appending a base prefix of 0[box] if base is 2, 8 or 16. */
PyAPI_FUNC(PyObject *) _PyLong_Format(PyObject *obj, int base);

/* For use by the gcd function in mathmodule.c */
PyAPI_FUNC(PyObject *) _PyLong_GCD(PyObject *, PyObject *);

PyAPI_FUNC(PyObject *) _PyLong_Rshift(PyObject *, size_t);
PyAPI_FUNC(PyObject *) _PyLong_Lshift(PyObject *, size_t);
