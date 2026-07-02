/* Abstract Object Interface (many thanks to Jim Fulton) */

#ifndef Py_ABSTRACTOBJECT_H
#define Py_ABSTRACTOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

/* === Object Protocol ================================================== */

/* Implemented elsewhere:

   int PyObject_Print(PyObject *o, FILE *fp, int flags);

   Print an object 'o' on file 'fp'.  Returns -1 on error. The flags argument
   is used to enable certain printing options. The only option currently
   supported is Py_Print_RAW.

   (What should be said about Py_Print_RAW?). */


/* Implemented elsewhere:

   int PyObject_HasAttrString(PyObject *o, const char *attr_name);

   Returns 1 if object 'o' has the attribute attr_name, and 0 otherwise.

   This is equivalent to the Python expression: hasattr(o,attr_name).

   This function always succeeds. */


/* Implemented elsewhere:

   PyObject* PyObject_GetAttrString(PyObject *o, const char *attr_name);

   Retrieve an attributed named attr_name form object o.
   Returns the attribute value on success, or NULL on failure.

   This is the equivalent of the Python expression: o.attr_name. */


/* Implemented elsewhere:

   int PyObject_HasAttr(PyObject *o, PyObject *attr_name);

   Returns 1 if o has the attribute attr_name, and 0 otherwise.

   This is equivalent to the Python expression: hasattr(o,attr_name).

   This function always succeeds. */

/* Implemented elsewhere:

   PyObject* PyObject_GetAttr(PyObject *o, PyObject *attr_name);

   Retrieve an attributed named 'attr_name' form object 'o'.
   Returns the attribute value on success, or NULL on failure.

   This is the equivalent of the Python expression: o.attr_name. */


/* Implemented elsewhere:

   int PyObject_SetAttrString(PyObject *o, const char *attr_name, PyObject *v);

   Set the value of the attribute named attr_name, for object 'o',
   to the value 'v'. Raise an exception and return -1 on failure; return 0 on
   success.

   This is the equivalent of the Python statement o.attr_name=v. */


/* Implemented elsewhere:

   int PyObject_SetAttr(PyObject *o, PyObject *attr_name, PyObject *v);

   Set the value of the attribute named attr_name, for object 'o', to the value
   'v'. an exception and return -1 on failure; return 0 on success.

   This is the equivalent of the Python statement o.attr_name=v. */

/* Implemented as a macro:

   int PyObject_DelAttrString(PyObject *o, const char *attr_name);

   Delete attribute named attr_name, for object o. Returns
   -1 on failure.

   This is the equivalent of the Python statement: del o.attr_name. */
#define PyObject_DelAttrString(O,A) PyObject_SetAttrString((O),(A), NULL)


/* Implemented as a macro:

   int PyObject_DelAttr(PyObject *o, PyObject *attr_name);

   Delete attribute named attr_name, for object o. Returns -1
   on failure.  This is the equivalent of the Python
   statement: del o.attr_name. */
#define  PyObject_DelAttr(O,A) PyObject_SetAttr((O),(A), NULL)


/* Implemented elsewhere:

   PyObject *PyObject_Repr(PyObject *o);

   Compute the string representation of object 'o'.  Returns the
   string representation on success, NULL on failure.

   This is the equivalent of the Python expression: repr(o).

   Called by the repr() built-in function. */


/* Implemented elsewhere:

   PyObject *PyObject_Str(PyObject *o);

   Compute the string representation of object, o.  Returns the
   string representation on success, NULL on failure.

   This is the equivalent of the Python expression: str(o).

   Called by the str() and print() built-in functions. */


/* Declared elsewhere

   PyAPI_FUNC(int) PyCallable_Check(PyObject *o);

   Determine if the object, o, is callable.  Return 1 if the object is callable
   and 0 otherwise.

   This function always succeeds. */


#ifdef PY_SSIZE_T_CLEAN
#  define PyObject_CallFunction _PyObject_CallFunction_SizeT
#  define PyObject_CallMethod _PyObject_CallMethod_SizeT
#endif


#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03090000
/* Call a callable Python object without any arguments */
PyAPI_FUNC(PyObject *) PyObject_CallNoArgs(PyObject *func);
#endif


/* Call a callable Python object 'callable' with arguments given by the
   tuple 'args' and keywords arguments given by the dictionary 'kwargs'.

   'args' must not be NULL, use an empty tuple if no arguments are
   needed. If no named arguments are needed, 'kwargs' can be NULL.

   This is the equivalent of the Python expression:
   callable(*args, **kwargs). */
PyAPI_FUNC(PyObject *) PyObject_Call(PyObject *callable,
                                     PyObject *args, PyObject *kwargs);


/* Call a callable Python object 'callable', with arguments given by the
   tuple 'args'.  If no arguments are needed, then 'args' can be NULL.

   Returns the result of the call on success, or NULL on failure.

   This is the equivalent of the Python expression:
   callable(*args). */
PyAPI_FUNC(PyObject *) PyObject_CallObject(PyObject *callable,
                                           PyObject *args);

/* Call a callable Python object, callable, with a variable number of C
   arguments. The C arguments are described using a mkvalue-style format
   string.

   The format may be NULL, indicating that no arguments are provided.

   Returns the result of the call on success, or NULL on failure.

   This is the equivalent of the Python expression:
   callable(arg1, arg2, ...). */
PyAPI_FUNC(PyObject *) PyObject_CallFunction(PyObject *callable,
                                             const char *format, ...);

/* Call the method named 'name' of object 'obj' with a variable number of
   C arguments.  The C arguments are described by a mkvalue format string.

   The format can be NULL, indicating that no arguments are provided.

   Returns the result of the call on success, or NULL on failure.

   This is the equivalent of the Python expression:
   obj.name(arg1, arg2, ...). */
PyAPI_FUNC(PyObject *) PyObject_CallMethod(PyObject *obj,
                                           const char *name,
                                           const char *format, ...);

PyAPI_FUNC(PyObject *) _PyObject_CallFunction_SizeT(PyObject *callable,
                                                    const char *format,
                                                    ...);

PyAPI_FUNC(PyObject *) _PyObject_CallMethod_SizeT(PyObject *obj,
                                                  const char *name,
                                                  const char *format,
                                                  ...);

/* Call a callable Python object 'callable' with a variable number of C
   arguments. The C arguments are provided as PyObject* values, terminated
   by a NULL.

   Returns the result of the call on success, or NULL on failure.

   This is the equivalent of the Python expression:
   callable(arg1, arg2, ...). */
PyAPI_FUNC(PyObject *) PyObject_CallFunctionObjArgs(PyObject *callable,
                                                    ...);

/* Call the method named 'name' of object 'obj' with a variable number of
   C arguments.  The C arguments are provided as PyObject* values, terminated
   by NULL.

   Returns the result of the call on success, or NULL on failure.

   This is the equivalent of the Python expression: obj.name(*args). */

PyAPI_FUNC(PyObject *) PyObject_CallMethodObjArgs(
    PyObject *obj,
    PyObject *name,
    ...);


/* Implemented elsewhere:

   Py_hash_t PyObject_Hash(PyObject *o);

   Compute and return the hash, hash_value, of an object, o.  On
   failure, return -1.

   This is the equivalent of the Python expression: hash(o). */


/* Implemented elsewhere:

   int PyObject_IsTrue(PyObject *o);

   Returns 1 if the object, o, is considered to be true, 0 if o is
   considered to be false and -1 on failure.

   This is equivalent to the Python expression: not not o. */


/* Implemented elsewhere:

   int PyObject_Not(PyObject *o);

   Returns 0 if the object, o, is considered to be true, 1 if o is
   considered to be false and -1 on failure.

   This is equivalent to the Python expression: not o. */


/* Get the type of an object.

   On success, returns a type object corresponding to the object type of object
   'o'. On failure, returns NULL.

   This is equivalent to the Python expression: type(o) */
PyAPI_FUNC(PyObject *) PyObject_Type(PyObject *o);


/* Return the size of object 'o'.  If the object 'o' provides both sequence and
   mapping protocols, the sequence size is returned.

   On error, -1 is returned.

   This is the equivalent to the Python expression: len(o) */
PyAPI_FUNC(Py_ssize_t) PyObject_Size(PyObject *o);


/* For DLL compatibility */
#undef PyObject_Length
PyAPI_FUNC(Py_ssize_t) PyObject_Length(PyObject *o);
#define PyObject_Length PyObject_Size

/* Return element of 'o' corresponding to the object 'key'. Return NULL
  on failure.

  This is the equivalent of the Python expression: o[key] */
PyAPI_FUNC(PyObject *) PyObject_GetItem(PyObject *o, PyObject *key);


/* Map the object 'key' to the value 'v' into 'o'.

   Raise an exception and return -1 on failure; return 0 on success.

   This is the equivalent of the Python statement: o[key]=v. */
PyAPI_FUNC(int) PyObject_SetItem(PyObject *o, PyObject *key, PyObject *v);

/* Remove the mapping for the string 'key' from the object 'o'.
   Returns -1 on failure.

   This is equivalent to the Python statement: del o[key]. */
PyAPI_FUNC(int) PyObject_DelItemString(PyObject *o, const char *key);

/* Delete the mapping for the object 'key' from the object 'o'.
   Returns -1 on failure.

   This is the equivalent of the Python statement: del o[key]. */
PyAPI_FUNC(int) PyObject_DelItem(PyObject *o, PyObject *key);


/* === Old Buffer API ============================================ */

/* FIXME:  usage of these should all be replaced in Python itself
   but for backwards compatibility we will implement them.
   Their usage without a corresponding "unlock" mechanism
   may create issues (but they would already be there). */

/* Takes an arbitrary object which must support the (character, single segment)
   buffer interface and returns a pointer to a read-only memory location
   usable as character based input for subsequent processing.

   Return 0 on success.  buffer and buffer_len are only set in case no error
   occurs. Otherwise, -1 is returned and an exception set. */
Py_DEPRECATED(3.0)
PyAPI_FUNC(int) PyObject_AsCharBuffer(PyObject *obj,
                                      const char **buffer,
                                      Py_ssize_t *buffer_len);

/* Checks whether an arbitrary object supports the (character, single segment)
   buffer interface.

   Returns 1 on success, 0 on failure. */
Py_DEPRECATED(3.0) PyAPI_FUNC(int) PyObject_CheckReadBuffer(PyObject *obj);

/* Same as PyObject_AsCharBuffer() except that this API expects (readable,
   single segment) buffer interface and returns a pointer to a read-only memory
   location which can contain arbitrary data.

   0 is returned on success.  buffer and buffer_len are only set in case no
   error occurs.  Otherwise, -1 is returned and an exception set. */
Py_DEPRECATED(3.0)
PyAPI_FUNC(int) PyObject_AsReadBuffer(PyObject *obj,
                                      const void **buffer,
                                      Py_ssize_t *buffer_len);

/* Takes an arbitrary object which must support the (writable, single segment)
   buffer interface and returns a pointer to a writable memory location in
   buffer of size 'buffer_len'.

   Return 0 on success.  buffer and buffer_len are only set in case no error
   occurs. Otherwise, -1 is returned and an exception set. */
Py_DEPRECATED(3.0)
PyAPI_FUNC(int) PyObject_AsWriteBuffer(PyObject *obj,
                                       void **buffer,
                                       Py_ssize_t *buffer_len);


/* === New Buffer API ============================================ */

/* Takes an arbitrary object and returns the result of calling
   obj.__format__(format_spec). */
PyAPI_FUNC(PyObject *) PyObject_Format(PyObject *obj,
                                       PyObject *format_spec);


/* ==== Iterators ================================================ */

/* Takes an object and returns an iterator for it.
   This is typically a new iterator but if the argument is an iterator, this
   returns itself. */
PyAPI_FUNC(PyObject *) PyObject_GetIter(PyObject *);

/* Takes an AsyncIterable object and returns an AsyncIterator for it.
   This is typically a new iterator but if the argument is an AsyncIterator,
   this returns itself. */
PyAPI_FUNC(PyObject *) PyObject_GetAIter(PyObject *);

/* Returns non-zero if the object 'obj' provides iterator protocols, and 0 otherwise.

   This function always succeeds. */
PyAPI_FUNC(int) PyIter_Check(PyObject *);

/* Returns non-zero if the object 'obj' provides AsyncIterator protocols, and 0 otherwise.

   This function always succeeds. */
PyAPI_FUNC(int) PyAIter_Check(PyObject *);

/* Takes an iterator object and calls its tp_iternext slot,
   returning the next value.

   If the iterator is exhausted, this returns NULL without setting an
   exception.

   NULL with an exception means an error occurred. */
PyAPI_FUNC(PyObject *) PyIter_Next(PyObject *);

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x030A0000

/* Takes generator, coroutine or iterator object and sends the value into it.
   Returns:
   - PYGEN_RETURN (0) if generator has returned.
     'result' parameter is filled with return value
   - PYGEN_ERROR (-1) if exception was raised.
     'result' parameter is NULL
   - PYGEN_NEXT (1) if generator has yielded.
     'result' parameter is filled with yielded value. */
PyAPI_FUNC(PySendResult) PyIter_Send(PyObject *, PyObject *, PyObject **);
#endif


/* === Number Protocol ================================================== */

/* Returns 1 if the object 'o' provides numeric protocols, and 0 otherwise.

   This function always succeeds. */
PyAPI_FUNC(int) PyNumber_Check(PyObject *o);

/* Returns the result of adding o1 and o2, or NULL on failure.

   This is the equivalent of the Python expression: o1 + o2. */
PyAPI_FUNC(PyObject *) PyNumber_Add(PyObject *o1, PyObject *o2);

/* Returns the result of subtracting o2 from o1, or NULL on failure.

   This is the equivalent of the Python expression: o1 - o2. */
PyAPI_FUNC(PyObject *) PyNumber_Subtract(PyObject *o1, PyObject *o2);

/* Returns the result of multiplying o1 and o2, or NULL on failure.

   This is the equivalent of the Python expression: o1 * o2. */
PyAPI_FUNC(PyObject *) PyNumber_Multiply(PyObject *o1, PyObject *o2);

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03050000
/* This is the equivalent of the Python expression: o1 @ o2. */
PyAPI_FUNC(PyObject *) PyNumber_MatrixMultiply(PyObject *o1, PyObject *o2);
#endif

/* Returns the result of dividing o1 by o2 giving an integral result,
   or NULL on failure.

   This is the equivalent of the Python expression: o1 // o2. */
PyAPI_FUNC(PyObject *) PyNumber_FloorDivide(PyObject *o1, PyObject *o2);

/* Returns the result of dividing o1 by o2 giving a float result, or NULL on
   failure.

   This is the equivalent of the Python expression: o1 / o2. */
PyAPI_FUNC(PyObject *) PyNumber_TrueDivide(PyObject *o1, PyObject *o2);

/* Returns the remainder of dividing o1 by o2, or NULL on failure.

   This is the equivalent of the Python expression: o1 % o2. */
PyAPI_FUNC(PyObject *) PyNumber_Remainder(PyObject *o1, PyObject *o2);

/* See the built-in function divmod.

   Returns NULL on failure.

   This is the equivalent of the Python expression: divmod(o1, o2). */
PyAPI_FUNC(PyObject *) PyNumber_Divmod(PyObject *o1, PyObject *o2);

/* See the built-in function pow. Returns NULL on failure.

   This is the equivalent of the Python expression: pow(o1, o2, o3),
   where o3 is optional. */
PyAPI_FUNC(PyObject *) PyNumber_Power(PyObject *o1, PyObject *o2,
                                      PyObject *o3);

/* Returns the negation of o on success, or NULL on failure.

 This is the equivalent of the Python expression: -o. */
PyAPI_FUNC(PyObject *) PyNumber_Negative(PyObject *o);

/* Returns the positive of o on success, or NULL on failure.

   This is the equivalent of the Python expression: +o. */
PyAPI_FUNC(PyObject *) PyNumber_Positive(PyObject *o);

/* Returns the absolute value of 'o', or NULL on failure.

   This is the equivalent of the Python expression: abs(o). */
PyAPI_FUNC(PyObject *) PyNumber_Absolute(PyObject *o);

/* Returns the bitwise negation of 'o' on success, or NULL on failure.

   This is the equivalent of the Python expression: ~o. */
PyAPI_FUNC(PyObject *) PyNumber_Invert(PyObject *o);

/* Returns the result of left shifting o1 by o2 on success, or NULL on failure.

   This is the equivalent of the Python expression: o1 << o2. */
PyAPI_FUNC(PyObject *) PyNumber_Lshift(PyObject *o1, PyObject *o2);

/* Returns the result of right shifting o1 by o2 on success, or NULL on
   failure.

   This is the equivalent of the Python expression: o1 >> o2. */
PyAPI_FUNC(PyObject *) PyNumber_Rshift(PyObject *o1, PyObject *o2);

/* Returns the result of bitwise and of o1 and o2 on success, or NULL on
   failure.

   This is the equivalent of the Python expression: o1 & o2. */
PyAPI_FUNC(PyObject *) PyNumber_And(PyObject *o1, PyObject *o2);

/* Returns the bitwise exclusive or of o1 by o2 on success, or NULL on failure.

   This is the equivalent of the Python expression: o1 ^ o2. */
PyAPI_FUNC(PyObject *) PyNumber_Xor(PyObject *o1, PyObject *o2);

/* Returns the result of bitwise or on o1 and o2 on success, or NULL on
   failure.

   This is the equivalent of the Python expression: o1 | o2. */
PyAPI_FUNC(PyObject *) PyNumber_Or(PyObject *o1, PyObject *o2);

/* Returns 1 if obj is an index integer (has the nb_index slot of the
   tp_as_number structure filled in), and 0 otherwise. */
PyAPI_FUNC(int) PyIndex_Check(PyObject *);

/* Returns the object 'o' converted to a Python int, or NULL with an exception
   raised on failure. */
PyAPI_FUNC(PyObject *) PyNumber_Index(PyObject *o);

/* Returns the object 'o' converted to Py_ssize_t by going through
   PyNumber_Index() first.

   If an overflow error occurs while converting the int to Py_ssize_t, then the
   second argument 'exc' is the error-type to return.  If it is NULL, then the
   overflow error is cleared and the value is clipped. */
PyAPI_FUNC(Py_ssize_t) PyNumber_AsSsize_t(PyObject *o, PyObject *exc);

/* Returns the object 'o' converted to an integer object on success, or NULL
   on failure.

   This is the equivalent of the Python expression: int(o). */
PyAPI_FUNC(PyObject *) PyNumber_Long(PyObject *o);

/* Returns the object 'o' converted to a float object on success, or NULL
  on failure.

  This is the equivalent of the Python expression: float(o). */
PyAPI_FUNC(PyObject *) PyNumber_Float(PyObject *o);


/* --- In-place variants of (some of) the above number protocol functions -- */

/* Returns the result of adding o2 to o1, possibly in-place, or NULL
   on failure.

   This is the equivalent of the Python expression: o1 += o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceAdd(PyObject *o1, PyObject *o2);

/* Returns the result of subtracting o2 from o1, possibly in-place or
   NULL on failure.

   This is the equivalent of the Python expression: o1 -= o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceSubtract(PyObject *o1, PyObject *o2);

/* Returns the result of multiplying o1 by o2, possibly in-place, or NULL on
   failure.

   This is the equivalent of the Python expression: o1 *= o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceMultiply(PyObject *o1, PyObject *o2);

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03050000
/* This is the equivalent of the Python expression: o1 @= o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceMatrixMultiply(PyObject *o1, PyObject *o2);
#endif

/* Returns the result of dividing o1 by o2 giving an integral result, possibly
   in-place, or NULL on failure.

   This is the equivalent of the Python expression: o1 /= o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceFloorDivide(PyObject *o1,
                                                   PyObject *o2);

/* Returns the result of dividing o1 by o2 giving a float result, possibly
   in-place, or null on failure.

   This is the equivalent of the Python expression: o1 /= o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceTrueDivide(PyObject *o1,
                                                  PyObject *o2);

/* Returns the remainder of dividing o1 by o2, possibly in-place, or NULL on
   failure.

   This is the equivalent of the Python expression: o1 %= o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceRemainder(PyObject *o1, PyObject *o2);

/* Returns the result of raising o1 to the power of o2, possibly in-place,
   or NULL on failure.

   This is the equivalent of the Python expression: o1 **= o2,
   or o1 = pow(o1, o2, o3) if o3 is present. */
PyAPI_FUNC(PyObject *) PyNumber_InPlacePower(PyObject *o1, PyObject *o2,
                                             PyObject *o3);

/* Returns the result of left shifting o1 by o2, possibly in-place, or NULL
   on failure.

   This is the equivalent of the Python expression: o1 <<= o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceLshift(PyObject *o1, PyObject *o2);

/* Returns the result of right shifting o1 by o2, possibly in-place or NULL
   on failure.

   This is the equivalent of the Python expression: o1 >>= o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceRshift(PyObject *o1, PyObject *o2);

/* Returns the result of bitwise and of o1 and o2, possibly in-place, or NULL
   on failure.

   This is the equivalent of the Python expression: o1 &= o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceAnd(PyObject *o1, PyObject *o2);

/* Returns the bitwise exclusive or of o1 by o2, possibly in-place, or NULL
   on failure.

   This is the equivalent of the Python expression: o1 ^= o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceXor(PyObject *o1, PyObject *o2);

/* Returns the result of bitwise or of o1 and o2, possibly in-place,
   or NULL on failure.

   This is the equivalent of the Python expression: o1 |= o2. */
PyAPI_FUNC(PyObject *) PyNumber_InPlaceOr(PyObject *o1, PyObject *o2);

/* Returns the integer n converted to a string with a base, with a base
   marker of 0b, 0o or 0x prefixed if applicable.

   If n is not an int object, it is converted with PyNumber_Index first. */
PyAPI_FUNC(PyObject *) PyNumber_ToBase(PyObject *n, int base);


/* === Sequence protocol ================================================ */

/* Return 1 if the object provides sequence protocol, and zero
   otherwise.

   This function always succeeds. */
PyAPI_FUNC(int) PySequence_Check(PyObject *o);

/* Return the size of sequence object o, or -1 on failure. */
PyAPI_FUNC(Py_ssize_t) PySequence_Size(PyObject *o);

/* For DLL compatibility */
#undef PySequence_Length
PyAPI_FUNC(Py_ssize_t) PySequence_Length(PyObject *o);
#define PySequence_Length PySequence_Size


/* Return the concatenation of o1 and o2 on success, and NULL on failure.

   This is the equivalent of the Python expression: o1 + o2. */
PyAPI_FUNC(PyObject *) PySequence_Concat(PyObject *o1, PyObject *o2);

/* Return the result of repeating sequence object 'o' 'count' times,
  or NULL on failure.

  This is the equivalent of the Python expression: o * count. */
PyAPI_FUNC(PyObject *) PySequence_Repeat(PyObject *o, Py_ssize_t count);

/* Return the ith element of o, or NULL on failure.

   This is the equivalent of the Python expression: o[i]. */
PyAPI_FUNC(PyObject *) PySequence_GetItem(PyObject *o, Py_ssize_t i);

/* Return the slice of sequence object o between i1 and i2, or NULL on failure.

   This is the equivalent of the Python expression: o[i1:i2]. */
PyAPI_FUNC(PyObject *) PySequence_GetSlice(PyObject *o, Py_ssize_t i1, Py_ssize_t i2);

/* Assign object 'v' to the ith element of the sequence 'o'. Raise an exception
   and return -1 on failure; return 0 on success.

   This is the equivalent of the Python statement o[i] = v. */
PyAPI_FUNC(int) PySequence_SetItem(PyObject *o, Py_ssize_t i, PyObject *v);

/* Delete the 'i'-th element of the sequence 'v'. Returns -1 on failure.

   This is the equivalent of the Python statement: del o[i]. */
PyAPI_FUNC(int) PySequence_DelItem(PyObject *o, Py_ssize_t i);

/* Assign the sequence object 'v' to the slice in sequence object 'o',
   from 'i1' to 'i2'. Returns -1 on failure.

   This is the equivalent of the Python statement: o[i1:i2] = v. */
PyAPI_FUNC(int) PySequence_SetSlice(PyObject *o, Py_ssize_t i1, Py_ssize_t i2,
                                    PyObject *v);

/* Delete the slice in sequence object 'o' from 'i1' to 'i2'.
   Returns -1 on failure.

   This is the equivalent of the Python statement: del o[i1:i2]. */
PyAPI_FUNC(int) PySequence_DelSlice(PyObject *o, Py_ssize_t i1, Py_ssize_t i2);

/* Returns the sequence 'o' as a tuple on success, and NULL on failure.

   This is equivalent to the Python expression: tuple(o). */
PyAPI_FUNC(PyObject *) PySequence_Tuple(PyObject *o);

/* Returns the sequence 'o' as a list on success, and NULL on failure.
   This is equivalent to the Python expression: list(o) */
PyAPI_FUNC(PyObject *) PySequence_List(PyObject *o);

/* Return the sequence 'o' as a list, unless it's already a tuple or list.

   Use PySequence_Fast_GET_ITEM to access the members of this list, and
   PySequence_Fast_GET_SIZE to get its length.

   Returns NULL on failure.  If the object does not support iteration, raises a
   TypeError exception with 'm' as the message text. */
PyAPI_FUNC(PyObject *) PySequence_Fast(PyObject *o, const char* m);

/* Return the size of the sequence 'o', assuming that 'o' was returned by
   PySequence_Fast and is not NULL. */
#define PySequence_Fast_GET_SIZE(o) \
    (PyList_Check(o) ? PyList_GET_SIZE(o) : PyTuple_GET_SIZE(o))

/* Return the 'i'-th element of the sequence 'o', assuming that o was returned
   by PySequence_Fast, and that i is within bounds. */
#define PySequence_Fast_GET_ITEM(o, i)\
     (PyList_Check(o) ? PyList_GET_ITEM(o, i) : PyTuple_GET_ITEM(o, i))

/* Return a pointer to the underlying item array for
   an object returned by PySequence_Fast */
#define PySequence_Fast_ITEMS(sf) \
    (PyList_Check(sf) ? ((PyListObject *)(sf))->ob_item \
                      : ((PyTupleObject *)(sf))->ob_item)

/* Return the number of occurrences on value on 'o', that is, return
   the number of keys for which o[key] == value.

   On failure, return -1.  This is equivalent to the Python expression:
   o.count(value). */
PyAPI_FUNC(Py_ssize_t) PySequence_Count(PyObject *o, PyObject *value);

/* Return 1 if 'ob' is in the sequence 'seq'; 0 if 'ob' is not in the sequence
   'seq'; -1 on error.

   Use __contains__ if possible, else _PySequence_IterSearch(). */
PyAPI_FUNC(int) PySequence_Contains(PyObject *seq, PyObject *ob);

/* For DLL-level backwards compatibility */
#undef PySequence_In
/* Determine if the sequence 'o' contains 'value'. If an item in 'o' is equal
   to 'value', return 1, otherwise return 0. On error, return -1.

   This is equivalent to the Python expression: value in o. */
PyAPI_FUNC(int) PySequence_In(PyObject *o, PyObject *value);

/* For source-level backwards compatibility */
#define PySequence_In PySequence_Contains


/* Return the first index for which o[i] == value.
   On error, return -1.

   This is equivalent to the Python expression: o.index(value). */
PyAPI_FUNC(Py_ssize_t) PySequence_Index(PyObject *o, PyObject *value);


/* --- In-place versions of some of the above Sequence functions --- */

/* Append sequence 'o2' to sequence 'o1', in-place when possible. Return the
   resulting object, which could be 'o1', or NULL on failure.

  This is the equivalent of the Python expression: o1 += o2. */
PyAPI_FUNC(PyObject *) PySequence_InPlaceConcat(PyObject *o1, PyObject *o2);

/* Repeat sequence 'o' by 'count', in-place when possible. Return the resulting
   object, which could be 'o', or NULL on failure.

   This is the equivalent of the Python expression: o1 *= count.  */
PyAPI_FUNC(PyObject *) PySequence_InPlaceRepeat(PyObject *o, Py_ssize_t count);


/* === Mapping protocol ================================================= */

/* Return 1 if the object provides mapping protocol, and 0 otherwise.

   This function always succeeds. */
PyAPI_FUNC(int) PyMapping_Check(PyObject *o);

/* Returns the number of keys in mapping object 'o' on success, and -1 on
  failure. This is equivalent to the Python expression: len(o). */
PyAPI_FUNC(Py_ssize_t) PyMapping_Size(PyObject *o);

/* For DLL compatibility */
#undef PyMapping_Length
PyAPI_FUNC(Py_ssize_t) PyMapping_Length(PyObject *o);
#define PyMapping_Length PyMapping_Size


/* Implemented as a macro:

   int PyMapping_DelItemString(PyObject *o, const char *key);

   Remove the mapping for the string 'key' from the mapping 'o'. Returns -1 on
   failure.

   This is equivalent to the Python statement: del o[key]. */
#define PyMapping_DelItemString(O,K) PyObject_DelItemString((O),(K))

/* Implemented as a macro:

   int PyMapping_DelItem(PyObject *o, PyObject *key);

   Remove the mapping for the object 'key' from the mapping object 'o'.
   Returns -1 on failure.

   This is equivalent to the Python statement: del o[key]. */
#define PyMapping_DelItem(O,K) PyObject_DelItem((O),(K))

/* On success, return 1 if the mapping object 'o' has the key 'key',
   and 0 otherwise.

   This is equivalent to the Python expression: key in o.

   This function always succeeds. */
PyAPI_FUNC(int) PyMapping_HasKeyString(PyObject *o, const char *key);

/* Return 1 if the mapping object has the key 'key', and 0 otherwise.

   This is equivalent to the Python expression: key in o.

   This function always succeeds. */
PyAPI_FUNC(int) PyMapping_HasKey(PyObject *o, PyObject *key);

/* On success, return a list or tuple of the keys in mapping object 'o'.
   On failure, return NULL. */
PyAPI_FUNC(PyObject *) PyMapping_Keys(PyObject *o);

/* On success, return a list or tuple of the values in mapping object 'o'.
   On failure, return NULL. */
PyAPI_FUNC(PyObject *) PyMapping_Values(PyObject *o);

/* On success, return a list or tuple of the items in mapping object 'o',
   where each item is a tuple containing a key-value pair. On failure, return
   NULL. */
PyAPI_FUNC(PyObject *) PyMapping_Items(PyObject *o);

/* Return element of 'o' corresponding to the string 'key' or NULL on failure.

   This is the equivalent of the Python expression: o[key]. */
PyAPI_FUNC(PyObject *) PyMapping_GetItemString(PyObject *o,
                                               const char *key);

/* Map the string 'key' to the value 'v' in the mapping 'o'.
   Returns -1 on failure.

   This is the equivalent of the Python statement: o[key]=v. */
PyAPI_FUNC(int) PyMapping_SetItemString(PyObject *o, const char *key,
                                        PyObject *value);

/* isinstance(object, typeorclass) */
PyAPI_FUNC(int) PyObject_IsInstance(PyObject *object, PyObject *typeorclass);

/* issubclass(object, typeorclass) */
PyAPI_FUNC(int) PyObject_IsSubclass(PyObject *object, PyObject *typeorclass);

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_ABSTRACTOBJECT_H
#  include "cpython/abstract.h"
#  undef Py_CPYTHON_ABSTRACTOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* Py_ABSTRACTOBJECT_H */
